import os
import sys
from email.policy import default


class Tool():
    # Nom affiché dans BioImageIT
    name = "Active Voxel Finder"

    # Description visible pour l'utilisateur
    description = "Apply thresholds to a 4D image sequence (T,Z,Y,X) to detect active voxels based on dynamic changes."

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
        dict(name='input_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path',
             autoColumn=True),
        dict(name='dynamic_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X) représentant les changements dynamiques.', required=True, type='Path'),
        dict(name='std_noise', help='Écart type du bruit pour le calcul du Z-score.', required=True, type='Float', default=1.1696291),
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', required=True, type='Path'),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', required=True, type='Path'),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='activeVoxels.tif',
             type='Path')
    ]

    def processAllData(self, argsList):
        """
        Traite toutes les données en des seuils pour détecter les voxels actifs dans une séquence d'images 4D.

        Paramètres :
            argsList : liste d'objets avec les attributs nécessaires pour chaque image

        Retour :
            None
        """

        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data
            from astroca.activeVoxels.activeVoxelsFinder import voxels_finder
        except ImportError as e:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'astroca'))

            if base_dir not in sys.path:
                sys.path.append(base_dir)

            try:
                import numpy as np
                from astroca.tools.loadData import load_data
                from astroca.tools.exportData import export_data
                from astroca.activeVoxels.activeVoxelsFinder import voxels_finder
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

        std_noise = float(argsList[0].std_noise)
        first_volume_dF = argsList[0].dynamic_image # Dynamic image for dF
        first_volume_dF = str(first_volume_dF)
        # Vérification du fichier d'entrée pour dF
        if not os.path.exists(first_volume_dF):
            raise FileNotFoundError(f"Le fichier d'entrée pour dF est introuvable : {first_volume_dF}")
        dF = load_data(first_volume_dF)
        # print(f"Shape of dF data: {dF.shape}")
        Z, Y, X = dF.shape
        dF4D = np.empty((time_length, Z, Y, X), dtype=dF.dtype)
        dF4D[0] = dF
        # Merge all the others input dF data into one 4D array
        for t, arg in enumerate(argsList[1:], start=1):
            input_path_dF = arg.dynamic_image
            input_path_dF = str(input_path_dF)

            # check files
            if not os.path.exists(input_path_dF):
                raise FileNotFoundError(f"Le fichier d'entrée pour dF est introuvable : {input_path_dF}")
            dF = load_data(input_path_dF)
            # print(f"Data shape for dF at time {t}: {dF.shape}")
            dF4D[t] = dF
        # print(f"Shape of merged dF data: {dF4D.shape}")

        output_image = argsList[0].output_image

        # Apply the active voxel finder
        processed_data = voxels_finder(
            data4D,
            dF4D,
            std_noise,
            index_xmin,
            index_xmax
        )

        # Save each time frame as a separate image
        file_name = str(os.path.basename(output_image))
        # remove .tif extension if present
        if file_name.endswith('.tif'):
            file_name = file_name[:-5]
        for t in range(time_length):
            file_name_t = f"{file_name}{t}.tif"
            data_to_export = processed_data[t][np.newaxis, ...]  # Add a new axis for time
            export_data(data_to_export, os.path.dirname(output_image), export_as_single_tif=True, file_name=file_name_t)




