import os
import sys
import subprocess

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
        dict(name='dynamic_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X) représentant les changements dynamiques.', required=True, type='Path', autoColumn=True),
        dict(name='std_noise', help='Chemin vers écart type du bruit pour le calcul du Z-score.', required=True, type='Path', autoColumn=True),
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', required=True, type='Path', autoColumn=True),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', required=True, type='Path', autoColumn=True),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='activeVoxels.tif',
             type='Path')
    ]

    def setup_environment(self):
        try:
            import astroca
            print("Package astroca déjà disponible")
            return
        except ImportError:
            print("Installation du package astroca depuis GitHub...")

        repo_url = "git+ssh://git@github.com/audigiem/AstrocytesSegmentation.git@bioimageIT_src"
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", repo_url
            ])
            print("Package astroca installé avec succès")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Échec de l'installation pip du package astroca : {e}")

    def processData(self, args):
        """
        Traite les données en appliquant la détection des voxels actifs.
        Paramètres :
            args : objet avec les attributs nécessaires pour l'image d'entrée, l'image dynamique, l'écart type du bruit,
                   et les indices xmin/xmax.
        """
        # Configuration de l'environnement
        self.setup_environment()

        # Import des modules après installation
        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data
            from astroca.activeVoxels.activeVoxelsFinder import voxels_finder
        except ImportError as e:
            raise ImportError("Impossible d'importer les modules nécessaires. "
                              "Vérifiez que le module 'astroca' est présent.") from e

        volume = str(args.input_image)
        # Vérification du fichier d'entrée
        if not os.path.exists(volume):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {volume}")
        data = load_data(volume)

        volume_dF = str(args.dynamic_image)
        if not os.path.exists(volume_dF):
            raise FileNotFoundError(f"Le fichier d'image dynamique est introuvable : {volume_dF}")
        data_dF = load_data(volume_dF)

        index_xmin_path = str(args.index_xmin)
        index_xmax_path = str(args.index_xmax)
        if not os.path.exists(index_xmin_path):
            raise FileNotFoundError(f"Le fichier d'indices xmin est introuvable : {index_xmin_path}")
        if not os.path.exists(index_xmax_path):
            raise FileNotFoundError(f"Le fichier d'indices xmax est introuvable : {index_xmax_path}")
        index_xmin = np.load(index_xmin_path)
        index_xmax = np.load(index_xmax_path)

        data_path = str(args.std_noise)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Le fichier de données est introuvable : {data_path}")
        # Load std_noise and mean_noise from the .txt file
        with open(data_path, 'r') as f:
            lines = f.readlines()
            _ = float(lines[0].strip())
            std_noise = float(lines[1].strip())

        output_image = args.output_image

        processed_data = voxels_finder(
            data,
            data_dF,
            std_noise,
            index_xmin,
            index_xmax
        )

        file_name = str(os.path.basename(output_image))
        # remove .tif extension if present
        if file_name.endswith('.tif'):
            file_name = file_name[:-4]
        export_data(processed_data, os.path.dirname(output_image), export_as_single_tif=True, file_name=file_name)

    def processAllData(self, argsList):
        for args in argsList:
            try:
                self.processData(args)
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {args.input_image}: {e}")
                continue




