import torch
import torch.nn as nn

from torchsummary import summary

import cfg

_namesofplaces = {#{{{
    'Conv': nn.Conv3d,
    'ConvTranspose': nn.ConvTranspose3d,
    'BatchNorm': nn.BatchNorm3d,
    'ReLU': nn.ReLU,
    'Softshrink': nn.Softshrink,
    'Hardshrink': nn.Hardshrink,
    'LeakyReLU': nn.LeakyReLU,
    'MSELoss': nn.MSELoss,
    'None': None,
    }
#}}}

def _merge(source, destination):#{{{
    # overwrites field in destination if field exists in source, otherwise just merges
    for key, value in source.items():
        if isinstance(value, dict):
            node = destination.setdefault(key, {})
            _merge(value, node)
        else:
            destination[key] = value
    return destination
#}}}

def __crop_tensor(x, w) :#{{{
    x = x.narrow(2,w/2,x.shape[2]-w)
    x = x.narrow(3,w/2,x.shape[3]-w)
    x = x.narrow(4,w/2,x.shape[4]-w)
    return x.contiguous()
#}}}

class BasicLayer(nn.Module) :#{{{
    __default_param = {#{{{
        'inplane': None,
        'outplane': None,

        'conv': 'Conv',
        'conv_kw': {
            'stride': 1,
            'padding': 1,
            'kernel_size': 3,
            'bias': True,
            },

        'batch_norm': 'BatchNorm',
        'batch_norm_kw': {
            'momentum': 0.1,
            },

        'activation': 'ReLU',
        'activation_kw': { },

        'crop_output': False,

        'dropout': False,
        'dropout_kw': {
            'p': 0.5,
            'inplace': False,
            },
        }
    #}}}
    def __init__(self, layer_dict) :#{{{
        super(BasicLayer, self).__init__()
        self.__merged_dict = _merge(
            layer_dict,
            copy.deepcopy(BasicLayer.__default_param)
            )

        if self.__merged_dict['conv'] is not None :
            self.__conv_fct = _namesofplaces[self.__merged_dict['conv']](
                self.__merged_dict['inplane'], self.__merged_dict['outplane'],
                **self.__merged_dict['conv_kw']
                )
        else :
            self.__conv_fct = nn.Identity()

        if self.__merged_dict['crop_output'] :
            self.__crop_fct = lambda x : __crop_tensor(x, self.__merged_dict['crop_output'])
        else :
            self.__crop_fct = nn.Identity()

        if self.__merged_dict['dropout'] :
            self.__dropout_fct = nn.Dropout3d(
                **self.__merged_dict['dropout_kw']
                )
        else :
            self.__dropout_fct = nn.Identity()

        if self.__merged_dict['batch_norm'] is not None :
            self.__batch_norm_fct = _namesofplaces[self.__merged_dict['batch_norm']](
                self.__merged_dict['outplane'],
                **self.__merged_dict['batch_norm_kw']
                )
        else :
            self.__batch_norm_fct = nn.Identity()
        
        if self.__merged_dict['activation'] is not None :
            self.__activation_fct = _namesofplaces[self.__merged_dict['activation']](
                **self.__merged_dict['activation_kw']
                )
        else :
            self.__activation_fct = nn.Identity()
    #}}}
    def forward(self, x) :#{{{
        x = self.__activation_fct(self.__batch_norm_fct(self.__dropout_fct(self.__crop_fct(self.__conv_fct(x)))))
        return x
    #}}}
#}}}

class Network(nn.Module) :#{{{
    def __init__(self, network_dict) :#{{{
        super(Network, self).__init__()
        self.network_dict = network_dict

        self.__blocks = nn.ModuleList()
        # even index blocks are in, odd are out, the last one is the bottom through block
        # last index is 2*(NLevels-1)
        for ii in range(self.network_dict['NLevels']-1) :
            if ii < self.network_dict['NLevels'] - 1 : # not in the bottom block
                self.__blocks.append(
                    Network.__feed_forward_block(
                        self.network_dict['Level_%d'%ii]['in']
                        )
                    )
                self.__blocks.append(
                    Network.__feed_forward_block(
                        self.network_dict['Level_%d'%ii]['out']
                        )
                    )
        self.__blocks.append(
            Network.__feed_forward_block(
                self.network_dict['Level_%d'%(self.network_dict['NLevels']-1)]['through']
                )
            )

        if 'feed_model' in self.network_dict :
            self.__feed_model = self.network_dict['feed_model']
        else :
            self.__feed_model = False

        if 'model_block' in self.network_dict :
            if not self.network_dict['feed_model'] :
                raise RuntimeError('You provided a model block but do not require model feed. Aborting.')
            self.__model_block = Network.__feed_forward_block(
                self.network_dict['model_block']
                )
        else :
            self.__model_block = None

        if 'globallocalskip' in self.network_dict :
            self.__globallocalskip = True
            self.__globallocalskip_feed_out = self.network_dict['globallocalskip']['feed_out']
            self.__globallocalskip_feed_in  = self.network_dict['globallocalskip']['feed_in']
            self.__globallocalskip_block    = Network.__feed_forward_block(
                self.network_dict['globallocalskip']['block']
                )
        else :
            self.__globallocalskip = False

        if 'multiply_model' in self.network_dict :
            self.__multiply_model = self.network_dict['multiply_model']
        else :
            self.__multiply_model = False

        if 'take_exponential' in self.network_dict :
            self.__take_exponential = self.network_dict['take_exponential']
        else :
            self.__take_exponential = False

        if 'take_sinh' in self.network_dict :   
            self.__take_sinh = self.network_dict['take_sinh']
        else :
            self.__take_sinh = False
        
        assert not (self.__take_exponential and self.__take_sinh), 'It does not make much sense do to both transformations.'

        self.is_frozen = False
    #}}}
    @staticmethod
    def __feed_forward_block(input_list) :#{{{
        layers = []
        for layer_dict in input_list :
            layers.append(BasicLayer(layer_dict))
        return nn.Sequential(*layers)
    #}}}
    def forward(self, x, xmodel) :#{{{
        intermediate_data = []
        xglobal = None

        # contracting path
        for ii in range(self.network_dict['NLevels']-1) :
            x = self.__blocks[2*ii](x)
            if self.network_dict['Level_%d'%ii]['concat'] :
                intermediate_data.append(x.clone())
            else :
                intermediate_data.append(None)

        # bottom level
        x = self.__blocks[2*(self.network_dict['NLevels']-1)](x)

        # expanding path
        for ii in range(self.network_dict['NLevels']-2, -1, -1) :
            if self.network_dict['Level_%d'%ii]['concat'] :
                if self.network_dict['Level_%d'%ii]['resize_to_gas'] :
                    intermediate_data[ii] = __crop_tensor(
                        intermediate_data[ii],
                        (intermediate_data[ii].shape[-1] * (cfg.DM_sidelength - cfg.gas_sidelength))/cfg.DM_sidelength
                        )
                x = torch.cat((x, intermediate_data[ii]), dim = 1)
            if self.__globallocalskip :
                if ii == self.__globallocalskip_feed_in :
                    x = torch.cat((x, xglobal), dim = 1)
            if ii == 0 and self.__take_exponential :
                x = torch.exp(x)
            if ii == 0 and self.__take_sinh :
                x = torch.sinh(x)
            if ii == 0 and self.__feed_model :
                if self.__model_block is not None :
                    xmodel = torch.cat((xmodel, self.__model_block(xmodel)), dim = 1)
                    # include a skip connection
                if not self.__multiply_model :
                    x = torch.cat((x, xmodel), dim = 1)
                else :
                    if cfg.dim == 1 :
                        x[:,0,...] = torch.mul(x[:,0,...], xmodel[:,0,...])
                    elif cfg.dim>1 and cfg.ftype in ['MOM', ] :
                        x[:,:cfg.dim,...] = torch.mul(x[:,:cfg.dim,...], xmodel)
                    elif cfg.dim>1 and cfg.ftype in ['MOM1', ] :
                        x[:,0,...] = torch.mul(x[:,0,...], xmodel[:,0,...])
            x = self.__blocks[2*ii+1](x)
            if self.__globallocalskip :
                if ii == self.__globallocalskip_feed_out :
                    xglobal = self.__globallocalskip_block(x)
        return x
    #}}}
#}}}


if __name__ == '__main__' :
    
    model = Network(cfg.this_network)

    # check that this is all correct
    summary(model, [(cfg.DM_sidelength, cfg.DM_sidelength, cfg.DM_sidelength, 1),
                    (cfg.gas_sidelength, cfg.gas_sidelength, cfg.gas_sidelength, 1)],
            device='cpu')
