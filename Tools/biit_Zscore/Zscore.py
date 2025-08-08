import os
import sys
import subprocess
class Tool():
    # Nom affiché dans BioImageIT
    name = "Z-score computation"
    
    # Description visible pour l'utilisateur
    description = "Inform on how much the data deviates from the mean, using the Z-score method."

    # Catégorie dans laquelle l'outil apparaîtra
    categories = ['Astroca', 'Active Voxel Detection']

    # Environnement conda spécifique
    environment = 'astroca-env'

    # Dépendances (tu peux adapter si besoin)
    dependencies = dict(
        python='==3.10',
        conda=['tqdm', 'numpy', 'pandas'],
        pip=[]
    )

    # Définition des entrées attendues
    inputs = [
        dict(name='input_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path', autoColumn=True),
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', required=True, type='Path', autoColumn=True),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', required=True, type='Path', autoColumn=True),
        dict(name='data', help='Moyenne du bruit et Écart-type pour la normalisation.', required=True, type='Path', autoColumn=True),
        dict(name='threshold', help='Seuil pour la détection des voxels actifs.', required=True, type='Float', default=2.8),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='zScore.tif', type='Path')
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
        Traite les données en appliquant la variance stabilisation.

        Paramètres :
            args : objet avec les attributs nécessaires pour l'image d'entrée et les indices xmin/xmax

        Retour :
            None
        """
        # Configuration de l'environnement
        self.setup_environment()

        # Import des modules après installation
        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data
            from astroca.activeVoxels.zScore import compute_z_score
        except ImportError as e:
            raise ImportError("Impossible d'importer les modules nécessaires. "
                              "Vérifiez que le module 'astroca' est présent.") from e

        image_path = str(args.input_image)  # Ensure it's a string path
        # Vérification du fichier d'entrée
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {image_path}")
        data = load_data(image_path)

        index_xmin_path = str(args.index_xmin)
        index_xmax_path = str(args.index_xmax)
        # Vérification des fichiers d'indices
        if not os.path.exists(index_xmin_path):
            raise FileNotFoundError(f"Le fichier index_xmin est introuvable : {index_xmin_path}")
        if not os.path.exists(index_xmax_path):
            raise FileNotFoundError(f"Le fichier index_xmax est introuvable : {index_xmax_path}")
        index_xmin = np.load(index_xmin_path)
        index_xmax = np.load(index_xmax_path)

        data_path = str(args.data)
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Le fichier de données est introuvable : {data_path}")
        # Load std_noise and mean_noise from the .txt file
        with open(data_path, 'r') as f:
            lines = f.readlines()
            mean_noise = float(lines[0].strip())
            std_noise = float(lines[1].strip())
        threshold = float(args.threshold)
        output_image = str(args.output_image)  # Ensure it's a string path

        # Apply the Z-score computation
        processed_data = compute_z_score(data, std_noise, mean_noise, threshold, index_xmin, index_xmax)

        # Save the processed data
        file_name = str(os.path.basename(output_image))
        # remove .tif extension if present
        if file_name.endswith('.tif'):
            file_name = file_name[:-4]
        export_data(processed_data, os.path.dirname(output_image), export_as_single_tif=True, file_name=file_name)
    
    def processAllData(self, argsList):
        if len(argsList) > 1:
            for args in argsList:
                try:
                    self.processData(args)
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {args.input_image}: {e}")
                    continue

