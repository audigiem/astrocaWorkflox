import os
import sys

class Tool():
    
    name = "Image amplitude estimator"
    description = "Estimates the amplitude with inverse Anscombe transform for a 4D image sequence (T,Z,Y,X)."
    categories = ['Astroca', 'Florescence Estimation']
    environment = 'astroca-env'

    dependencies = dict(
        python='==3.10',
        conda=['tqdm', 'numpy', 'pandas', 'numba'],
        pip=[]
    )

    inputs = [
        dict(name='input_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path', autoColumn=True),
        dict(name='f0_image', help='Chemin vers le fichier .tif contenant l\'estimation du fond (F0).', required=True, type='Path', autoColumn=True),
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', required=True, type='Path'),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', required=True, type='Path'),
    ]

    outputs = [
        dict(name='output_image', help='The output image.', 
             default='inverse_anscombe_transformed_volume.tif', type='Path')
    ]

    def processAllData(self, argsList):
        """ Apply inverse Anscombe transform to compute the amplitude of the image sequence. """
        # print(f"DEBUG: Current working directory: {os.getcwd()}")
        # print(f"DEBUG: __file__ location: {__file__}")
        # print(f"DEBUG: sys.path: {sys.path[:3]}...")  # Afficher les 3 premiers éléments
        
        try:
            import numpy as np
            # print("DEBUG: numpy imported successfully")
            from astroca.tools.loadData import load_data
            # print("DEBUG: load_data imported successfully")
            from astroca.tools.exportData import export_data
            # print("DEBUG: export_data imported successfully")
            from astroca.dynamicImage.dynamicImage import compute_image_amplitude
            # print("DEBUG: background_estimation_single_block imported successfully")
        except ImportError as e:
            # print(f"DEBUG: First import failed: {e}")
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'astroca'))
            # print(f"DEBUG: Trying to add to path: {base_dir}")
            # print(f"DEBUG: Directory exists: {os.path.exists(base_dir)}")
            # if os.path.exists(base_dir):
                # print(f"DEBUG: Contents of {base_dir}: {os.listdir(base_dir)[:5]}...")  # Afficher les 5 premiers éléments
            
            if base_dir not in sys.path:
                sys.path.append(base_dir)
                # print(f"DEBUG: Added {base_dir} to sys.path")
            # else:
                # print(f"DEBUG: {base_dir} already in sys.path")
                
            try:
                import numpy as np
                # print("DEBUG: numpy imported successfully (second try)")
                from astroca.tools.loadData import load_data
                # print("DEBUG: load_data imported successfully (second try)")
                from astroca.tools.exportData import export_data
                # print("DEBUG: export_data imported successfully (second try)")
                from astroca.dynamicImage.dynamicImage import compute_image_amplitude
                # print("DEBUG: background_estimation_single_block imported successfully (second try)")
            except ImportError as e:
                # print(f"DEBUG: Second import also failed: {e}")
                
                # Essayons une approche différente pour trouver astroca
                possible_paths = [
                    os.path.join(os.path.dirname(__file__), '..', '..', 'astroca'),
                    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'astroca'),
                    os.path.join(os.path.dirname(__file__), 'astroca'),
                    '/home/matteo/Bureau/INRIA/codePython/astroca',  # Chemin absolu basé sur votre structure
                ]
                
                for path in possible_paths:
                    abs_path = os.path.abspath(path)
                    # print(f"DEBUG: Checking path: {abs_path}")
                    if os.path.exists(abs_path):
                        # print(f"DEBUG: Found astroca at: {abs_path}")
                        if abs_path not in sys.path:
                            sys.path.append(abs_path)
                        try:
                            from astroca.tools.loadData import load_data
                            from astroca.tools.exportData import export_data
                            from astroca.dynamicImage.dynamicImage import compute_image_amplitude
                            # print("DEBUG: Successfully imported astroca modules!")
                            break
                        except ImportError as e2:
                            # print(f"DEBUG: Import failed even with path {abs_path}: {e2}")
                            continue
                else:
                    raise ImportError("Impossible d'importer les modules nécessaires. "
                                    "Vérifiez que le module 'astroca' est présent.") from e

        # Le reste du code reste identique
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
            
        f0_image = str(argsList[0].f0_image)
        if not os.path.exists(f0_image):
            raise FileNotFoundError(f"Fichier F0 introuvable : {f0_image}")
        f0_data = load_data(f0_image)
        f0_data = f0_data[np.newaxis, ...]  # Ajouter une dimension pour le temps

        xmin_path = argsList[0].index_xmin
        xmax_path = argsList[0].index_xmax
        xmin_path = str(xmin_path)
        xmax_path = str(xmax_path)
        # we have xmin_path = ..../index_xmin0.npy lets remove the 0 at the end
        if xmin_path.endswith('0.npy'):
            xmin_path = xmin_path[:-5] + '.npy'
        if xmax_path.endswith('0.npy'):
            xmax_path = xmax_path[:-5] + '.npy'
        if not os.path.exists(xmin_path):
            raise FileNotFoundError(f"Le fichier index_xmin est introuvable : {xmin_path}")
        if not os.path.exists(xmax_path):
            raise FileNotFoundError(f"Le fichier index_xmax est introuvable : {xmax_path}")
        index_xmin = np.load(xmin_path)
        index_xmax = np.load(xmax_path)
        output_image = str(argsList[0].output_image)

        param_amplitude = {
            'files': {'save_results': 0},
            'paths': {'output_dir': None}
        }

        processed_data = compute_image_amplitude(
            data4D, f0_data, index_xmin, index_xmax, param_amplitude
        )

        # Save each time frame as a separate image
        file_name = str(os.path.basename(output_image))
        # remove .tif extension if present
        if file_name.endswith('.tif'):
            file_name = file_name[:-5]
        for t in range(time_length):
            file_name_t = f"{file_name}{t}.tif"
            data_to_export = processed_data[t][np.newaxis, ...]  # Add a new axis for time
            # print(f"Export Data shape: {data_to_export.shape}, File Name: {file_name_t}")
            export_data(data_to_export, os.path.dirname(output_image), export_as_single_tif=True, file_name=file_name_t)