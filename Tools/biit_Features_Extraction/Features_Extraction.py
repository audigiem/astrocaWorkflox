import os
import sys
from email.policy import default


class Tool():
    # Nom affiché dans BioImageIT
    name = "Features Extraction"

    # Description visible pour l'utilisateur
    description = "This tool extracts features from a 4D image sequence (T,Z,Y,X) using various methods."

    # Catégorie dans laquelle l'outil apparaîtra
    categories = ['Astroca', 'Features']

    # Environnement conda spécifique
    environment = 'astroca-env'

    # Dépendances (tu peux adapter si besoin)
    dependencies = dict(
        python='==3.10',
        conda=['tqdm', 'numpy', 'pandas', 'openpyxl'],
        pip=[]
    )

    # Définition des entrées attendues
    inputs = [
        dict(name='input_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path',
             autoColumn=True),
        dict(name='image_amplitude', help='Chemin vers le fichier .tif 4D (T,Z,Y,X) représentant l\'amplitude de l\'image.', required=True, type='Path'),
        dict(name='ids_events', help='Identifiants des événements détectés (de 1 à ids_events)', required=True, type='int', default=6),
        dict(name='voxel_size_x', help='Taille du voxel en X en µm', required=True, type='Float', default=0.1025),
        dict(name='voxel_size_y', help='Taille du voxel en Y en µm', required=True, type='Float', default=0.1025),
        dict(name='voxel_size_z', help='Taille du voxel en Z en µm', required=True, type='Float', default=0.1344),
        dict(name='threshold_median_localized', help='Seuil de la médiane localisée pour la détection des caractéristiques.', required=True, type='Float', default=4.0),
        dict(name='threshold_distance_localized', help='Seuil de la distance localisée pour la détection des caractéristiques.', required=True, type='Float', default=6.0),
        dict(name='volume_localized', help='Volume localisé pour la détection des caractéristiques.', required=True, type='Float', default=0.0434),
    ]

    outputs = [
        dict(name='features', help='Caractéristiques extraites de l\'image.', default='features_extracted.csv',
             type='Path')
    ]

    def processAllData(self, argsList):
        """
        Traite toutes les données en extrayant les caractéristiques d'une séquence d'images 4D.
        """
        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data
            from astroca.features.featuresComputation import save_features_from_events
        except ImportError as e:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'astroca'))
           
            if base_dir not in sys.path:
                sys.path.append(base_dir)
            
            try:
                import numpy as np
                from astroca.tools.loadData import load_data
                from astroca.tools.exportData import export_data
                from astroca.features.featuresComputation import save_features_from_events
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
                            from astroca.features.featuresComputation import save_features_from_events

                            break
                        except ImportError as e2:

                            continue
                else:
                    raise ImportError("Impossible d'importer les modules nécessaires. "
                                      "Vérifiez que le module 'astroca' est présent.") from e

        time_length = len(argsList)
        first_volume = argsList[0].input_image
        first_volume = str(first_volume)  # Ensure it's a string path
        # Vérification du fichier d'entrée
        if not os.path.exists(first_volume):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {first_volume}")
        data = load_data(first_volume)
        # print(f"Shape of loaded data: {data.shape}")

        Z, Y, X = data.shape
        data4D = np.empty((time_length, Z, Y, X), dtype=data.dtype)
        data4D[0] = data  # Initialize the first time frame

        # Merge all the others input data into one 4D array
        for t, arg in enumerate(argsList[1:], start=1):
            input_path = arg.input_image
            input_path = str(input_path)

            # check files
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {input_path}")
            data = load_data(input_path)
            # print(f"Data shape for time {t}: {data.shape}")
            data4D[t] = data

        # print(f"Shape of merged data: {data4D.shape}")
        
        # Load image amplitude
        first_volume_amplitude = argsList[0].image_amplitude
        first_volume_amplitude = str(first_volume_amplitude)
        # Vérification du fichier d'entrée pour l'amplitude
        if not os.path.exists(first_volume_amplitude):
            raise FileNotFoundError(f"Le fichier d'entrée pour l'amplitude est introuvable : {first_volume_amplitude}")
        image_amplitude = load_data(first_volume_amplitude)
        # print(f"Shape of loaded image amplitude: {image_amplitude.shape}")
        Z, Y, X = image_amplitude.shape
        image_amplitude_4D = np.empty((time_length, Z, Y, X), dtype=image_amplitude.dtype)
        image_amplitude_4D[0] = image_amplitude
        # Merge all the others input image amplitude data into one 4D array
        for t, arg in enumerate(argsList[1:], start=1):
            input_path_amplitude = arg.image_amplitude
            input_path_amplitude = str(input_path_amplitude)

            # check files
            if not os.path.exists(input_path_amplitude):
                raise FileNotFoundError(f"Le fichier d'entrée pour l'amplitude est introuvable : {input_path_amplitude}")
            image_amplitude = load_data(input_path_amplitude)
            # print(f"Image amplitude shape for time {t}: {image_amplitude.shape}")
            image_amplitude_4D[t] = image_amplitude
        # print(f"Shape of merged image amplitude data: {image_amplitude_4D.shape}")
        
        # load other parameters
        ids_events = int(argsList[0].ids_events)
        voxel_size_x = float(argsList[0].voxel_size_x)
        voxel_size_y = float(argsList[0].voxel_size_y)
        voxel_size_z = float(argsList[0].voxel_size_z)
        threshold_median_localized = float(argsList[0].threshold_median_localized)
        threshold_distance_localized = float(argsList[0].threshold_distance_localized)
        volume_localized = float(argsList[0].volume_localized)
        
        output_feature = argsList[0].features

        
        
        # Prepare parameters for feature extraction
        param_features_extraction = {
            'features_extraction': {
                'ids_events': ids_events,
                'voxel_size_x': voxel_size_x,
                'voxel_size_y': voxel_size_y,
                'voxel_size_z': voxel_size_z,
                'threshold_median_localized': threshold_median_localized,
                'threshold_distance_localized': threshold_distance_localized,
                'volume_localized': volume_localized
            },
            'files': {'save_results': 1},
            'paths': {'output_dir': os.path.dirname(output_feature)+"/"}
        }


        save_features_from_events(data4D, ids_events, image_amplitude_4D, param_features_extraction)

        




