import numpy as np
import os
from collections import OrderedDict
import pickle
import json

# Utility classes for wind farm datasets.

# Each farm dataset should have a correspondence between its own channels and the universal channel names.
# When a channel is completely missing, a NaN is entered in its returned values.
UNIVERSAL_COL_NAMES= ['mean_wsp','mean_power', 'cos_phi', 'cos_yaw','sin_yaw', 'rated_power','rotor_diam','ambient_temp','wsp_std','is_masked']

# colnames:                        [mean wsp], [P], [cosPhi], [cos_yaw] [sin yaw] [rated_power/2300], [rotor_diam/93] []
#node_max_vals_lillgrund = np.array([30.,       2341., 1.,      1.,       1. ,       2300.,            93,             146.16,1.])

# Max normalization for all channels:
MAX_NORM_DICT = OrderedDict([
    ('mean_wsp',30.),
    ('mean_power', 2300.),
    ('cos_phi',1.),
    ('cos_yaw',1.),
    ('sin_yaw',1.),
    ('rated_power', 2300),
    ('rotor_diam',93.),
    ('ambient_temp',146.),
    ('wsp_std',1.),
    ('is_masked',1.)])

MAX_NORM_DICT = OrderedDict([
    ('mean_wsp',30. * 0.185),
    ('mean_power', 5000.*0.39),
    ('cos_phi',1.),
    ('cos_yaw',1.),
    ('sin_yaw',1.),
    ('rated_power', 5000),
    ('rotor_diam',120.),
    ('ambient_temp',146.),
    ('wsp_std',1.),
    ('is_masked',1.)])
node_max_vals = np.array([v for v in MAX_NORM_DICT.values()])

def _node_normalizer(node_vals):
    node_vals = node_vals/node_max_vals
    return node_vals

def _node_unnormalizer(node_vals):
    node_vals = node_vals * node_max_vals
    return node_vals


def _global_normalizer(x):
    """
    First two dimensions are angles, no need to normalize.
    Third dimension is max windspeed -> normalization by 
    dividing with a constant number (same for all farms)
    """
    x[:,-1] = x[:,-1]/10 # a "global" windspeed
    return x

def _global_unnormalizer(x):
    x[:,1] = x[:,-1]*10.
    return node_vals


def _edge_normalizer(edge_data):
    edge_data[:,2] = edge_data[:,2]/400
    return edge_data

def _compute_static_edge_features(X):
    """
    Computes a [nturb, nturb, 3] matrix that contains edge 
    features for all turbines in the farm. The features are
    [sin(p) cos(p) , D] where "p" is the angle defined 
    between the turbines and "D" is the distance between the 
    turbines. 
    """
    nfeats = 3
    feats = np.zeros([X.shape[0],X.shape[0], nfeats])

    # for every turbine:
    for t1 in range(X.shape[0]):
        d_xy = X[t1] - X
        phi = np.arctan2(d_xy[:,1],d_xy[:,0])
        phi[t1] = 0 # self-edge
        sp, cp = [f(phi) for f in [np.sin, np.cos]]
        d = np.sqrt(np.sum(d_xy**2,1))
        feats[t1,:,:] = np.array([sp,cp,d]).T
    for t1 in range(X.shape[0]):
        feats[t1,t1,:] = 0.

    return feats

def _compute_edge_features(X, nac_angle, act_power, angle_range_cutoff = -0.9):
    """
    (UNUSED)
    Computes features for edges of a farm-graph (modeling wake interactions btw turbines).
    Returns two features that depend on the power of the up-stream turbine, the distance between the
    turbines and the alignment of the wind of the up-wind turbine w.r.t. the line defined by each
    pair of turbines.

    parameters:
      X         :  [nturbs, 2] "X" and "Y" coordinates of turbines (assumed in the same plane).
      nac_angle :  the nacelle angle from north (?)
      act_power :  the power of all turbines. It affects the edges
      angle_range_cutoff : [-0.5] the value of the inner product of the turbine alignment vector
                    and the up-wind turbine wind orientation to be kept.
    """
    feats = compute_static_edge_features(X)

    cs, ss = [feats[:,:,1] ,feats[:,:,0]] # <- the cos and sin of every turbine with all other turbines
    DD = (nac_angle)/360*(2*np.pi) # <- in degrees for Lillgrund
    PP = act_power
    cs_, ss_ = [f(DD) for f in [np.cos, np.sin]]

    # The following computes the inner product of the upstream turbine yaw angle and
    # the vector defined btwn the two turbines: (checked visually - results seem ok)
    c_ = np.einsum('ji, jk -> ijk', cs, cs_)
    s_ = np.einsum('ji, jk -> ijk', ss, ss_)
    # V = (c_ + s_)*PP/2000 # If this is negative, the wind points away from the line determined by i,j.
    V1 = (c_ + s_)
    V2 = V1*PP
    return V1,V2, feats[:,:,0], feats[:,:,1] , feats[:,:,2]


class FarmSCADADataset:
    def __init__(self, *args, **kwargs):
        self._name_ = "Unknown"
        self.has_stats = False
        if 'name' in kwargs.keys():
            self._name_ = kwargs['name']

    def _repr_html_(self):
        self.node_feats
        s = ''
        s += "<h1>%s farm dataset</h1>"%(self._name_)
        s += "<p>Dataset with %i turbines and %i timesteps (SCADA data snapshots).</p>"%(self.nturbines,self.ntimesteps)
        s += "<table>"
        
        if self.has_stats:
            s += "<tr> <th> Name </th> <th> Min </th> <th> Max </th> <th> Std </th> </tr>"
            for name, min_, max_, std_ in zip(UNIVERSAL_COL_NAMES, self._min, self._max, self._std):
                s += "<tr>" + "<td> %s </td>"%name + " ".join(["<td>%2.3f </td>"%v for v in [min_, max_, std_]]) + "</tr>"

            s+= "</table>"
        s += '<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This dataset and pre-processing code are licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.'
        s += '<br> </br>'

        return s


class FLORISFarm(FarmSCADADataset):
    def __init__(self, dataroot =['SCADA' ,'SimulatedFarm_01'], adj_cutoff_factor = 7):
        super(FLORISFarm, self).__init__(name = "FLORIS Simulation")
        self.data_root = os.path.join(*dataroot)
        with open(os.path.join(*dataroot, 'FarmData.npy'),'rb') as f:
            fdat = pickle.load(f)
        with open(os.path.join(*dataroot,'Xsim.npy'),'rb') as f:
            data = pickle.load(f)
        self.fdat = fdat
        self.data = data
        self.data[:,1,:] = self.data[:,1,:]/1000. # From Watts to kWatts 
        self.X = fdat['X']
        self.nturbines = self.X.shape[0]
        self.edge_features_all_turbs = _compute_static_edge_features(self.X)

        with open(os.path.join(*dataroot,'floris_input.json'),'r') as f:
            input_dict = json.load(f)

        self.turb_rotor_diameter = input_dict['turbine']['properties']['rotor_diameter']
        self.rated_power = 5000.

        self._adj_turb_rotor_cutoff_factor = adj_cutoff_factor
        self._prepare_data()
        self._min, self._max, self._std = [np.nanmin(np.nanmin(self.node_feats,1),1),np.nanmax(np.nanmax(self.node_feats,1),1), np.nanmean(np.nanstd(self.node_feats,2),1)]
        self.has_stats = True

    def _repr_html_(self):
        s = super()._repr_html_()
        s+= '<div> <b> Simulation dataset root folder: </b> %s </div>'%(self.data_root)
        return s

    def _prepare_data(self):
        self.universal_colnames_to_colnames = OrderedDict(
                [('mean_wsp','mean_wsp'),
                 ('mean_power','power'),
                 ('sin_yaw','sin_yaw'),
                 ('rotor_diam','rotor_diam'),
                 ('rated_power', 'rated_power'), 
                 ('cos_yaw','cos_yaw'),
                 ('wsp_std','turb_intensity')])
        self.feature_names = [n for n in self.universal_colnames_to_colnames.values()]
        
        self.ntimesteps = self.data.shape[-1]

        self.missing_columns = [c for c in UNIVERSAL_COL_NAMES if c not in self.universal_colnames_to_colnames.keys()]
        
        node_feats = [];

        self.node_feat_names = []
        for c in UNIVERSAL_COL_NAMES:
            if c == 'rated_power':
                node_feats.append(
                          np.ones([self.nturbines, self.ntimesteps])*self.rated_power
                        )
                self.node_feat_names.append('rated_power')
                continue
            if c == 'rotor_diam':
                node_feats.append(
                          np.ones([self.nturbines, self.ntimesteps])*self.turb_rotor_diameter
                        )
                self.node_feat_names.append('rotor_diam')
                continue

            if c in self.missing_columns:
                node_feats.append(np.zeros([self.nturbines, self.ntimesteps]))
            else:
                cn = self.fdat['chan_names'].index(self.universal_colnames_to_colnames[c])
                node_feats.append(self.data[:,cn,:])
            self.node_feat_names.append(c)

        self.node_feats = np.dstack(node_feats).transpose([2,0,1])
        Wsp = self.node_feats[0,:,:]
        global_wsp_max = np.nanmax(Wsp,0)
        global_wsp_min = np.nanmin(Wsp,0)
#
        ss = self.node_feats[3,:,:]
        sc = self.node_feats[4,:,:]
        global_ang_vars  = np.dstack([ss, sc])
#
        self.global_ang_vars  = np.nanmean(global_ang_vars, 0)
        self.global_wsp_max = global_wsp_max
#
        self.global_cond_vars = np.hstack([self.global_ang_vars, self.global_wsp_max[:,np.newaxis]])
        adjacency_cutoff_thresh  = self._adj_turb_rotor_cutoff_factor * self.turb_rotor_diameter #turb_rotor_diameter.flatten()[0]
 
        ###############################################
        # Edge features based cut-off
        edge_feats = self.edge_features_all_turbs
        val= edge_feats[:,:,-1]
        adj_cutoff = val < adjacency_cutoff_thresh
        self.from_nodes, self.to_nodes = np.where(adj_cutoff)
#
        self.edge_feats = np.repeat(edge_feats[self.from_nodes, self.to_nodes][..., np.newaxis], self.node_feats.shape[-1], axis = -1).transpose([1,0,2])
        self.edge_static_prop_inds = [0,1,2]
        self.node_static_prop_inds = [5,6]


    def sample_graphs(self, ngraphs = None, idx = None,
                      node_normalizer = _node_normalizer,
                      global_normalizer = _global_normalizer,
                      edge_normalizer = _edge_normalizer,
                      return_normalized_nodes = True,
                      return_normalized_edges = True,
                      return_normalized_global = True,
                      return_conditioning_separately = True,
                      random_rotations = False,
                      return_inds = False):
        """
        Samples randomly indices from the dataset and returns graph data representing the farm state.
        returns:
          gt            : GraphTuple object (input of the AE)
          node_nan_mask : node nan mask
          node_static_props : node channels that are static properties (i.e. turbine rated power, rotor diam)
          edge_static_props : edge channels that are static properties (i.e. distance, cos/sin of angle btwn turbines)

        """
        nfeats_nodes = self.node_feats.shape[0]
        nfeats_edges = self.edge_feats.shape[0]

        if idx is not None:
            if ngraphs is not None:
                print("Warning: you defined both ngraph and idx parameter! Are you sure you know what you are doing with this sampling function?")
            ngraphs = len(idx)

        node_shift_idx = self.nturbines

        from_nodes_tot = []
        to_nodes_tot   = []
        for nn in range(ngraphs):
            from_nodes_tot.extend(np.array(self.from_nodes)+node_shift_idx* nn)
            to_nodes_tot.extend(np.array(self.to_nodes)+node_shift_idx * nn)

        if idx is None:
            idx = np.random.choice(self.node_feats.shape[-1],ngraphs, replace = False)

        global_attr = self.global_cond_vars[idx].copy()
        node_data = np.stack([n[..., idx] for n in self.node_feats]).transpose([2,1,0]).reshape([-1,nfeats_nodes])
        #edge_data = self.edge_feats[:,:,idx].reshape([ nfeats_edges,-1]).T
        edge_data = np.stack([e[...,idx] for e in self.edge_feats]).transpose([2,1,0]).reshape([-1,nfeats_edges])

        if random_rotations:
            p = np.random.rand(1)*np.pi*2
            r = np.array([[np.cos(p),np.sin(p)],[-np.sin(p), np.cos(p)]])[:,:,0]
            edge_data[:,0:2] = edge_data[:,0:2] @ r
            node_data[:,3:5] = node_data[:,3:5] @ r
            global_attr[:,0:2] = global_attr[:,0:2] @ r




        # Keep track of NaNs for ignoring them in training.
        node_nan_mask = np.isnan(node_data)
        node_data[node_nan_mask] = 0.


        if return_normalized_nodes:
            node_data = node_normalizer(node_data)

        if return_normalized_edges:
            edge_data = edge_normalizer(edge_data)

        if return_normalized_global:
            global_attr = global_normalizer(global_attr)

        #gt = GraphTuple(node_data,
        #                edge_data,
        #                from_nodes_tot,
         #               to_nodes_tot,
         #               n_nodes=[self.nturbines]*ngraphs ,
         #               n_edges=[len(self.from_nodes)]*ngraphs)

        gt_dict = {
                'nodes'     : node_data,  
                'edges'     : edge_data,
                'senders'   : from_nodes_tot,
                'receivers' : to_nodes_tot,
                'n_nodes'   : [self.nturbines]*ngraphs,
                'n_edges'   : [len(self.from_nodes)] * ngraphs
                }

        edge_static_props = gt_dict['edges'][:,self.edge_static_prop_inds]
        node_static_props = gt_dict['nodes'][:,self.node_static_prop_inds]

        if not return_inds:
            return gt_dict, node_nan_mask , node_static_props, edge_static_props, global_attr
        else:
            return gt_dict, node_nan_mask , node_static_props, edge_static_props, global_attr, idx

