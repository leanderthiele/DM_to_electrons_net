
# input, output are scalars
dim = 1

# size of the dark matter input boxes
DM_sidelength = 64

# size of the gas output boxes
gas_sidelength = 32

# type of the output field (should not be important as long as 1-d fields are used)
ftype = 'N'

# our pretty confusing representation of the network
this_network = {#{{{
    'NLevels': 8,
    'feed_model': True,
    'take_sinh': True,

    'Level_0': {
        'concat': False,
        'in': [
                {
                    'inplane': 1,
                    'outplane': 32,
                },
        ],
        'out': [ # gets 32^3 input
                { # Final layer, collapse to single channel
                    'inplane': 64,
                    'outplane': 1,
                    'conv_kw': {
                                    'stride': 1,
                                    'kernel_size': 1,
                                    'padding': 0,
                               },
                    'batch_norm': None,
                },
        ],
    },

    'Level_1': {
        'concat': True,
        'resize_to_gas': True,
        'in': [
                {
                    'inplane': 32,
                    'outplane': 32,
                },
                {
                    'inplane': 32,
                    'outplane': 32,
                },
                {
                    'inplane': 32,
                    'outplane': 32,
                },
        ],
        'out': [
                {
                    'inplane': 64,
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 63, # remove one channel to make space for model
                },
        ],
    },

    'Level_2': {
        'concat': True,
        'resize_to_gas': False,
        'in': [
                {
                    'inplane': 32,
                    'outplane': 64,
                    'conv_kw': {
                                    'stride': 2,
                               },
                },
                {
                    'inplane': 64,
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 64,
                },
        ],
        'out': [
                {
                    'inplane': 128,
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 64,
                },
                {
                    'inplane': 64,
                    'outplane': 32,
                },
        ],
    },

    'Level_3': {
        'concat': True,
        'resize_to_gas': False,
        'in': [
                {
                    'inplane': 64,
                    'outplane': 128,
                    'conv_kw': {
                                    'stride': 2,
                               },
                    'dropout_kw': {
                                    'p': 0.2,
                                  },
                },
                {
                    'inplane': 128,
                    'outplane': 128,
                    'dropout_kw': {
                                    'p': 0.2,
                                  },
                },
                {
                    'inplane': 128,
                    'outplane': 128,
                    'dropout_kw': {
                                    'p': 0.2,
                                  },
                },
        ],
        'out': [
                {
                    'inplane':  256,
                    'outplane': 128,
                    'dropout_kw': {
                                    'p': 0.2,
                                  },
                },
                {
                    'inplane': 128,
                    'outplane': 128,
                    'dropout_kw': {
                                    'p': 0.2,
                                  },
                },
                {
                    'inplane': 128,
                    'outplane': 64,
                    'conv': 'ConvTranspose',
                    'conv_kw': {
                                    'stride': 2,
                                    'padding': 0,
                               },
                    'crop_output': 1,
                    'dropout_kw': {
                                    'p': 0.2,
                                  },
                },
        ],
    },

    'Level_4': {
        'concat': True,
        'resize_to_gas': False,
        'in': [
                {
                    'inplane': 128,
                    'outplane': 256,
                    'conv_kw': {
                                    'stride': 2,
                               },
                    'dropout_kw': {
                                    'p': 0.3,
                                  },
                },
                {
                    'inplane': 256,
                    'outplane': 256,
                    'dropout_kw': {
                                    'p': 0.3,
                                  },
                },
                {
                    'inplane': 256,
                    'outplane': 256,
                    'dropout_kw': {
                                    'p': 0.3,
                                  },
                },
        ],
        'out': [
                {
                    'inplane': 512,
                    'outplane': 256,
                    'dropout_kw': {
                                    'p': 0.3,
                                  },
                },
                {
                    'inplane': 256,
                    'outplane': 256,
                    'dropout_kw': {
                                    'p': 0.3,
                                  },
                },
                {
                    'inplane': 256,
                    'outplane': 128,
                    'conv': 'ConvTranspose',
                    'conv_kw': {
                                    'stride': 2,
                                    'padding': 0,
                               },
                    'crop_output': 1,
                    'dropout_kw': {
                                    'p': 0.3,
                                  },
                },
        ],
    },

    'Level_5': {
        'concat': True,
        'resize_to_gas': False,
        'in': [
                {
                    'inplane': 256,
                    'outplane': 512,
                    'conv_kw': {
                                    'stride': 2,
                               },
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.4,
                                  },
                },
                {
                    'inplane': 512,
                    'outplane': 512,
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.4,
                                  },
                },
                {
                    'inplane': 512,
                    'outplane': 512,
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.4,
                                  },
                },
        ],
        'out': [
                {
                    'inplane': 1024,
                    'outplane': 512,
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.4,
                                  },
                },
                {
                    'inplane': 512,
                    'outplane': 512,
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.4,
                                  },
                },
                {
                    'inplane': 512,
                    'outplane': 256,
                    'conv': 'ConvTranspose',
                    'conv_kw': {
                                    'stride': 2,
                                    'padding': 0,
                               },
                    'crop_output': 1,
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.4,
                                  },
                },
        ],
    },

    'Level_6': {
        'concat': True,
        'resize_to_gas': False,
        'in': [
                {
                    'inplane': 512,
                    'outplane': 1024,
                    'conv_kw': {
                                    'stride': 2,
                               },
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.5,
                                  },
                },
                {
                    'inplane': 1024,
                    'outplane': 1024,
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.5,
                                  },
                },
                {
                    'inplane': 1024,
                    'outplane': 1024,
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.5,
                                  },
                },
        ],
        'out': [
                {
                    'inplane': 2048,
                    'outplane': 1024,
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.5,
                                  },
                },
                {
                    'inplane': 1024,
                    'outplane': 1024,
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.5,
                                  },
                },
                {
                    'inplane': 1024,
                    'outplane': 512,
                    'conv': 'ConvTranspose',
                    'conv_kw': {
                                    'stride': 2,
                                    'padding': 0,
                               },
                    'crop_output': 1,
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.5,
                                  },
                },
        ],
    },

    'Level_7': {
        'through': [
                {
                    'inplane': 1024,
                    'outplane': 2048,
                    'conv_kw': {
                                    'stride': 2,
                               },
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.7,
                                  },
                },
                {
                    'inplane': 2048,
                    'outplane': 2048,
                    'conv_kw': {
                                    'stride': 1,
                                    'kernel_size': 1,
                                    'padding': 0,
                               },
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.7,
                                  },
                },
                {
                    'inplane': 2048,
                    'outplane': 2048,
                    'conv_kw': {
                                    'stride': 1,
                                    'kernel_size': 1,
                                    'padding': 0,
                               },
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.7,
                                  },
                },
                {
                    'inplane': 2048,
                    'outplane': 1024,
                    'conv': 'ConvTranspose',
                    'conv_kw': {
                                    'stride': 2,
                                    'padding': 0,
                               },
                    'crop_output': 1,
                    'dropout': True,
                    'batch_norm': None,
                    'dropout_kw': {
                                    'p': 0.7,
                                  },
                },
        ],
    },
}#}}}
