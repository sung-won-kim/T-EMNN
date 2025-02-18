import torch
import pickle
import torch.nn as nn
import lightning.pytorch as pl
from sklearn.metrics import r2_score
from torch.utils.data import Dataset
from torch.nn import Linear, Sequential, ReLU
from layers import Thickness_ProcessorLayer as ProcessorLayer

class LargeDataset(Dataset):
    def __init__(self, dataset_files, basepath, args):
        self.dataset_files = dataset_files
        self.basepath = basepath
        self.args = args

    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, idx):
        file_path = self.dataset_files[idx]
        with open(f"{self.basepath}/{file_path}", "rb") as f:
            data = pickle.load(f)

        data.shape_id = file_path[:-4]
        return data

class T_EMNN(pl.LightningModule):
    def __init__(self, raw_sample_data, args):
        super(T_EMNN, self).__init__()
        self.args = args

        input_dim_node = raw_sample_data.node_attr.shape[1]
        cond_dim = raw_sample_data.conds_feat.shape[1]
        input_dim_edge = raw_sample_data.edge_attr.shape[1]
        output_dim = raw_sample_data.y.shape[1]
        coord_dim_node = raw_sample_data.x.shape[1]

        self.node_encoder = Sequential(Linear(input_dim_node, args.hidden_dim),
                            ReLU(),
                            Linear(args.hidden_dim, args.hidden_dim)
                            )

        self.edge_encoder = Sequential(Linear( input_dim_edge, args.hidden_dim),
                            ReLU(),
                            Linear( args.hidden_dim, args.hidden_dim)
                            )
        
        self.processor = nn.ModuleList()
        assert (self.args.num_layers >= 1), 'Number of message passing layers is not >=1'

        processor_layer=self.build_processor_model()
        for _ in range(self.args.num_layers):
            self.processor.append(processor_layer(args.hidden_dim, args.hidden_dim))

        self.thick_processor = nn.ModuleList()
        for _ in range(self.args.num_layers):
            self.thick_processor.append(processor_layer(args.hidden_dim, args.hidden_dim))

        self.cond_encoder = Sequential(Linear(cond_dim, args.hidden_dim),
                            ReLU(),
                            Linear(args.hidden_dim, args.hidden_dim)
                            )

        self.decoder = Sequential(Linear( 2* args.hidden_dim , args.hidden_dim),
                            ReLU(),
                            Linear( args.hidden_dim, output_dim),
                            )

        self.coord_encoder = Sequential(Linear(coord_dim_node, args.hidden_dim),
                            ReLU(),
                            Linear(args.hidden_dim, args.hidden_dim)
                            )

        self.combine_decoder = Sequential(Linear(2 * args.hidden_dim, args.hidden_dim),
                            ReLU(),
                            Linear(args.hidden_dim, args.hidden_dim),
                            )
        
        self.thick_edge_encoder = Sequential(Linear(2, args.hidden_dim),
                            ReLU(),
                            Linear(args.hidden_dim, args.hidden_dim),
                            )
        
        self.thick_threshold = nn.Parameter(torch.tensor([0.0]))
    
    def get_lr(self, optimizer):
        return [group['lr'] for group in optimizer.param_groups]
    
    def build_processor_model(self):
        return ProcessorLayer

    def inverse_transform(self, transformed_points, rotation_matrix, center_mass):
        inverse_rotation = rotation_matrix[0].T
        inverse_rotation = torch.FloatTensor(inverse_rotation).to(transformed_points.device)
        original_points_centered = transformed_points @ inverse_rotation
        original_points = original_points_centered + torch.FloatTensor(center_mass[0]).to(transformed_points.device)
        return original_points
    
    def loss(self, pred, inputs):
        labels = inputs.y

        mae=torch.mean(torch.abs(labels-pred))
        error=torch.sum((labels-pred)**2,axis=1)
        loss=torch.sqrt(torch.mean(error)) ## RMSE
        r2 = r2_score(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())
        return loss, mae, r2
    
    def forward(self, data):
        node_feat = data.node_attr
        edge_feat = data.edge_attr
        edge_index = data.edge_index
        inv_coord = data.inv_x

        h = self.node_encoder(node_feat) 
        edge_h = self.edge_encoder(edge_feat)
        
        thick_edge_attr = data.thick_edge_attr
        thick_edge_attr[:,0] = torch.relu(thick_edge_attr[:,0])
        thick_edge_attr[:,1] = torch.log(thick_edge_attr[:,1] + 1e-6)
        thick_edge_h = self.thick_edge_encoder(thick_edge_attr)

        thick_threshold = self.thick_threshold
        mask = torch.sigmoid(-self.args.alpha * (data.thick_edge_attr[:,1] - thick_threshold))
        thick_edge_weight = mask.unsqueeze(1)

        for i in range(self.args.num_layers):
            h, edge_h = self.processor[i](h, edge_index, edge_h)
            h, thick_edge_h = self.thick_processor[i](h, data.thick_edge_index, thick_edge_h, thick_edge_weight)
            
        h_coord = self.coord_encoder(inv_coord)
        h = torch.cat([h, h_coord], dim=1)
        h = self.combine_decoder(h)

        cond_feat = data.conds_feat
        cond_h = self.cond_encoder(cond_feat)

        h = torch.cat([h, cond_h], dim=1)
        
        pred = self.decoder(h)
        pred = self.inverse_transform(pred, data.rotation_matrix, data.center_mass)

        return pred, [thick_edge_weight]
        
    def training_step(self, batch, batch_idx):
        pred, utils = self(batch)
        thick_edge_weight = utils[0]

        current_lr = self.get_lr(self.trainer.optimizers[0])

        sparsity_loss = -torch.mean(torch.abs(thick_edge_weight))

        rmse, mae, r2 = self.loss(pred, batch)

        self.log("Train RMSE", rmse, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Train MAE", mae, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Train R2", r2, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Thick Threshold", torch.nn.functional.softplus(self.thick_threshold).item(), prog_bar=True, batch_size=self.args.batch_size)
        self.log('lr', current_lr[0], prog_bar=True)  
        self.log('t_lr', current_lr[1], prog_bar=True)
        
        loss = rmse + 0.001 * sparsity_loss
            
        return loss

    def validation_step(self, batch, batch_idx):
        pred, _ = self(batch)

        rmse, mae, r2 = self.loss(pred, batch)

        self.log("Valid RMSE", rmse, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
        self.log("Valid MAE", mae, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Valid R2", r2, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        pred, _ = self(batch)

        rmse, mae, r2 = self.loss(pred, batch)

        self.log("Test RMSE", rmse, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)
        self.log("Test MAE", mae, prog_bar=True, batch_size=self.args.batch_size)
        self.log("Test R2", r2, prog_bar=True, batch_size=self.args.batch_size, sync_dist=True, on_epoch=True)

    def configure_optimizers(self):
        excluded_modules = ['thick_threshold']

        model_params = []
        model_param_names = []

        for name, param in self.named_parameters():
            if not any(name.startswith(excluded) for excluded in excluded_modules):
                model_params.append(param)
                model_param_names.append(name)

        optimizer = torch.optim.Adam(
            [
                {'params': model_params, 'lr': self.args.lr, 'weight_decay': self.args.weight_decay},
                {'params': [self.thick_threshold], 'lr': self.args.thres_lr},
            ],
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=5,
            threshold=1,
            threshold_mode='rel',
            verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Thick Threshold",
                "interval": "epoch",
                "reduce_on_plateau": True,
            },
        }

    def lr_scheduler_step(self, scheduler, metric):
        """Custom step to schedule only the second param group."""
        if scheduler is not None and metric is not None:
            scheduler.step(metric)
            scheduler.optimizer.param_groups[0]['lr'] = self.args.lr
