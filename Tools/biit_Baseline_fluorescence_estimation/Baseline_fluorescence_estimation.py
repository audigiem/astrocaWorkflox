import os
import sys

class Tool():
    
    name = "Background estimator"
    description = "This tool estimates the background of a 4D image sequence (T,Z,Y,X) using a moving window approach."
    categories = ['Astroca', 'Florescence Estimation']
    environment = 'astroca-env'

    dependencies = dict(
        python='==3.10',
        conda=['tqdm', 'numpy', 'pandas', 'numba'],
        pip=[]
    )

    inputs = [
        dict(name='input_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path', autoColumn=True),
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', required=True, type='Path'),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', required=True, type='Path'),
        dict(name='moving_window', help="Window size for background estimation.", required=False, type='Int', default=2),
    ]

    outputs = [
        dict(name='output_image', help='The output image.', 
             default='F0_estimated.tif', type='Path')
    ]

    def processAllData(self, argsList):
        """ Apply background estimation on the input image sequence. """
        
        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data
            from astroca.dynamicImage.dynamicImage import background_estimation_single_block
        except ImportError as e:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'astroca'))
            
            if base_dir not in sys.path:
                sys.path.append(base_dir)
                
            try:
                import numpy as np
                from astroca.tools.loadData import load_data
                from astroca.tools.exportData import export_data
                from astroca.dynamicImage.backgroundEstimator import background_estimation_single_block
            except ImportError as e:
                
                possible_paths = [
                    os.path.join(os.path.dirname(__file__), '..', '..', 'astroca'),
                    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'astroca'),
                    os.path.join(os.path.dirname(__file__), 'astroca'),
                    '/home/matteo/Bureau/INRIA/codePython/astroca',  # Chemin absolu basé sur votre structure
                ]
                
                for path in possible_paths:
                    abs_path = os.path.abspath(path)
                    if os.path.exists(abs_path):
                        if abs_path not in sys.path:
                            sys.path.append(abs_path)
                        try:
                            from astroca.tools.loadData import load_data
                            from astroca.tools.exportData import export_data
                            from astroca.dynamicImage.backgroundEstimator import background_estimation_single_block
                            break
                        except ImportError as e2:
                            continue
                else:
                    raise ImportError("Impossible d'importer les modules nécessaires. "
                                    "Vérifiez que le module 'astroca' est présent.") from e

        time_length = len(argsList)
        first_volume = str(argsList[0].input_image)

        if not os.path.exists(first_volume):
            raise FileNotFoundError(f"Fichier d'entrée introuvable : {first_volume}")

        data = load_data(first_volume)
        Z, Y, X = data.shape
        data4D = np.empty((time_length, Z, Y, X), dtype=data.dtype)
        data4D[0] = data

        for t, arg in enumerate(argsList[1:], start=1):
            input_path = str(arg.input_image)
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Fichier d'entrée introuvable : {input_path}")
            data = load_data(input_path)
            data4D[t] = data
            
        xmin_path = str(argsList[0].index_xmin)
        xmax_path = str(argsList[0].index_xmax)
        if xmin_path.endswith('0.npy'):
            xmin_path = xmin_path[:-5] + '.npy'
        if xmax_path.endswith('0.npy'):
            xmax_path = xmax_path[:-5] + '.npy'
        if not os.path.exists(xmin_path):
            raise FileNotFoundError(f"Fichier index_xmin introuvable : {xmin_path}")    
        if not os.path.exists(xmax_path):
            raise FileNotFoundError(f"Fichier index_xmax introuvable : {xmax_path}")
        
        xmin = np.load(xmin_path)
        xmax = np.load(xmax_path)
        output_image = str(argsList[0].output_image)
        moving_window = argsList[0].moving_window

        param_background_estimation = {
            'background_estimation': {
                'moving_window': moving_window,
                'method': 'percentile',
                'method2': 'Med',
                'percentile': 10,
            },
            'files': {'save_results': 0},
            'paths': {'output_dir': None}
        }

        processed_data = background_estimation_single_block(
            data4D, xmin, xmax, param_background_estimation
        )

        file_name = str(os.path.basename(output_image))
        data_to_export = processed_data[0][np.newaxis, ...]
        export_data(data_to_export, os.path.dirname(output_image), export_as_single_tif=True, file_name=file_name)