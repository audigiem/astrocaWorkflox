import os
import sys
from email.policy import default


class Tool():
    # Nom affiché dans BioImageIT
    name = "Calcium Active Voxel Finder"

    # Description visible pour l'utilisateur
    description = "Apply a connex component analysis to a 4D image sequence (T,Z,Y,X) to detect active voxels based on dynamic changes."

    # Catégorie dans laquelle l'outil apparaîtra
    categories = ['Astroca', 'Event Detection']

    # Environnement conda spécifique
    environment = 'astroca-env'

    # Dépendances (tu peux adapter si besoin)
    dependencies = dict(
        python='==3.10',
        conda=['tqdm', 'numpy', 'pandas', 'numba', 'matplotlib'],
        pip=[]
    )

    # Définition des entrées attendues
    inputs = [
        dict(name='input_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path',
             autoColumn=True),
        dict(name='threshold_size_3d', help='Taille minimale des composants connexes en 3D pour être considérées actives.',
             default=400, type='Integer', autoColumn=True),
        dict(name='threshold_correlation', help='Seuil de corrélation pour détecter les changements dynamiques.',
             default=0.6, type='Float', autoColumn=True),
        dict(name='threshold_size_3d_remove',
             help='Taille minimale des composants connexes en 3D pour être retirées de la détection.',
             default=20, type='Integer', autoColumn=True),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='calciumEvents.tif',
             type='Path'),
        dict(name='ids_events', help='Identifiants des événements détectés (de 1 à ids_events)', type='Integer', autoColumn=True)
    ]

    def processAllData(self, argsList):
        """
        Traite toutes les données en des seuils pour détecter les voxels actifs dans une séquence d'images 4D.

        Paramètres :
            argsList : liste d'objets avec les attributs nécessaires pour chaque image

        Retour :
            None
        """
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
            from astroca.events.eventDetectorCorrected import detect_calcium_events_opti
            # print("DEBUG: closing_morphology_in_space imported successfully")
        except ImportError as e:
            # print(f"DEBUG: First import failed: {e}")
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'astroca'))
            # print(f"DEBUG: Trying to add to path: {base_dir}")
            # print(f"DEBUG: Directory exists: {os.path.exists(base_dir)}")
            # if os.path.exists(base_dir):
            #     print(
            #         f"DEBUG: Contents of {base_dir}: {os.listdir(base_dir)[:5]}...")  # Afficher les 5 premiers éléments

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
                from astroca.events.eventDetectorCorrected import detect_calcium_events_opti
                # print("DEBUG: detect_calcium_events_opti imported successfully (second try)")
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
                            from astroca.dynamicImage.backgroundEstimator import background_estimation_single_block
                            # print("DEBUG: Successfully imported astroca modules!")
                            break
                        except ImportError as e2:
                            # print(f"DEBUG: Import failed even with path {abs_path}: {e2}")
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

        threshold_size_3d = int(argsList[0].threshold_size_3d)
        threshold_correlation = float(argsList[0].threshold_correlation)
        threshold_size_3d_remove = int(argsList[0].threshold_size_3d_remove)

        output_image = argsList[0].output_image
        
        param_event_finder = {
            'events_extraction' : {
                'threshold_size_3d': threshold_size_3d,
                'threshold_corr': threshold_correlation,
                'threshold_size_3d_removed': threshold_size_3d_remove
            },
            'files' : {'save_results': 0},
            'paths' : {'output_dir': None}
        }

        # Apply the active voxel finder
        processed_data, ids_events = detect_calcium_events_opti(data4D, param_event_finder)

        # Save each time frame as a separate image
        file_name = str(os.path.basename(output_image))
        # remove .tif extension if present
        if file_name.endswith('.tif'):
            file_name = file_name[:-5]
        for t in range(time_length):
            file_name_t = f"{file_name}{t}.tif"
            data_to_export = processed_data[t][np.newaxis, ...]  # Add a new axis for time
            export_data(data_to_export, os.path.dirname(output_image), export_as_single_tif=True, file_name=file_name_t)

        output_ids_events = int(ids_events)
        self.outputs[1]['ids_events'] = output_ids_events
        print(f"DEBUG: Number of detected events: {output_ids_events}")


