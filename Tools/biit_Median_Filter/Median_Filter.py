import os
import sys


class Tool():
    # Nom affiché dans BioImageIT
    name = "3D Median filter"

    # Description visible pour l'utilisateur
    description = "Apply a 3D median filter to a 4D image sequence (T,Z,Y,X) to reduce noise while preserving edges."

    # Catégorie dans laquelle l'outil apparaîtra
    categories = ['Astroca', 'Active Voxel Detection']

    # Environnement conda spécifique
    environment = 'astroca-env'

    # Dépendances (tu peux adapter si besoin)
    dependencies = dict(
        python='==3.10',
        conda=['tqdm', 'numpy', 'pandas', 'scipy', 'scikit-image', 'numba'],
        pip=[]
    )

    # Définition des entrées attendues
    inputs = [
        dict(name='closed_data', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path',
             autoColumn=True),
        dict(name='radius', help='Rayon pour l\'opération de fermeture.', required=True, type='Float', default=1.5),
        dict(name='border_mode', help='Mode de gestion des bords (reflect, constant, etc.).', required=False,
             type='Str', default='ignore'),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='medianFiltered.tif',
             type='Path')
    ]

    def processAllData(self, argsList):
        """
        Traite toutes les données en appliquant un filtre médian 3D.

        Paramètres :
            argsList : liste d'objets avec les attributs nécessaires pour chaque image

        Retour :
            None
        """
        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data
            from astroca.activeVoxels.medianFilter import unified_median_filter_3d
        except ImportError as e:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'astroca'))
            
            if base_dir not in sys.path:
                sys.path.append(base_dir)
               
            try:
                import numpy as np
                from astroca.tools.loadData import load_data
                from astroca.tools.exportData import export_data
                from astroca.activeVoxels.medianFilter import unified_median_filter_3d
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
                            from astroca.activeVoxels.medianFilter import unified_median_filter_3d
                            break
                        except ImportError as e2:
                            continue
                else:
                    raise ImportError("Impossible d'importer les modules nécessaires. "
                                      "Vérifiez que le module 'astroca' est présent.") from e

        time_length = len(argsList)
        first_volume = argsList[0].closed_data
        first_volume = str(first_volume)  # Ensure it's a string path
        # Vérification du fichier d'entrée
        if not os.path.exists(first_volume):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {first_volume}")
        data = load_data(first_volume)
        print(f"Shape of loaded data: {data.shape}")

        Z, Y, X = data.shape
        data4D = np.empty((time_length, Z, Y, X), dtype=data.dtype)
        data4D[0] = data  # Initialize the first time frame

        # Merge all the others input data into one 4D array
        for t, arg in enumerate(argsList[1:], start=1):
            input_path = arg.closed_data
            input_path = str(input_path)

            # check files
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {input_path}")
            data = load_data(input_path)
            print(f"Data shape for time {t}: {data.shape}")
            data4D[t] = data

        print(f"Shape of merged data: {data4D.shape}")

        # Load xmin and xmax indices
        radius = float(argsList[0].radius)
        border_mode = str(argsList[0].border_mode)

        output_image = argsList[0].output_image

        # Apply the space closing operation
        processed_data = unified_median_filter_3d(data4D, radius, border_mode)

        # Save each time frame as a separate image
        file_name = str(os.path.basename(output_image))
        # remove .tif extension if present
        if file_name.endswith('.tif'):
            file_name = file_name[:-5]
        for t in range(time_length):
            file_name_t = f"{file_name}{t}.tif"
            data_to_export = processed_data[t][np.newaxis, ...]  # Add a new axis for time
            export_data(data_to_export, os.path.dirname(output_image), export_as_single_tif=True, file_name=file_name_t)




