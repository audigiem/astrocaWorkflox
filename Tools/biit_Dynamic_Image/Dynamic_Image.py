import os
import sys
import subprocess
class Tool():
    # Nom affiché dans BioImageIT
    name = "Dynamic Image computation"
    
    # Description visible pour l'utilisateur
    description = "This tool computes the dynamic image of a 4D sequence (T,Z,Y,X) by substracting the mean image from each time frame."
    
    # Catégorie dans laquelle l'outil apparaîtra
    categories = ['Astroca', 'Florescence Estimation']

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
        dict(name='background_image', help='Chemin vers le fichier .tif contenant l\'image de fond.', required=True, type='Path', autoColumn=True),
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', required=True, type='Path', autoColumn=True),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', required=True, type='Path', autoColumn=True),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='dynamic_image_dF.tif', type='Path'),
        dict(name='output_data', help='Moyenne du bruit pour la normalisation et écart type.', default='data.txt', type='Path'),
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
            from astroca.dynamicImage.dynamicImage import compute_dynamic_image
            from astroca.parametersNoise.parametersNoise import estimate_std_over_time, estimate_std_over_time_optimized
        except ImportError as e:
            raise ImportError("Impossible d'importer les modules nécessaires. "
                              "Vérifiez que le module 'astroca' est présent.") from e


        first_volume = str(args.input_image)

        # Vérification du fichier d'entrée
        if not os.path.exists(first_volume):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {first_volume}")

        data = load_data(first_volume)
        T, _, _, _ = data.shape

        index_xmin_path = str(args.index_xmin)
        index_xmax_path = str(args.index_xmax)
        # Vérification des fichiers d'indices
        if not os.path.exists(index_xmin_path):
            raise FileNotFoundError(f"Le fichier index_xmin est introuvable : {index_xmin_path}")
        if not os.path.exists(index_xmax_path):
            raise FileNotFoundError(f"Le fichier index_xmax est introuvable : {index_xmax_path}")
        index_xmin = np.load(index_xmin_path)
        index_xmax = np.load(index_xmax_path)

        f0_path = str(args.background_image)
        # Vérification du fichier de fond
        if not os.path.exists(f0_path):
            raise FileNotFoundError(f"Le fichier de fond est introuvable : {f0_path}")
        dataF0 = load_data(f0_path)
        if dataF0.ndim == 3:
            # If dataF0 is 3D, we need to expand it to 4D by adding a new axis for time
            dataF0 = dataF0[np.newaxis, ...]

        output_image = str(args.output_image)
        output_data = str(args.output_data)

        param_dynamicImage = {
            'save': {'save_df': 0},
            'paths': {'output_dir': None}
        }

        processed_data, mean_noise = compute_dynamic_image(
            data,
            dataF0,
            index_xmin,
            index_xmax,
            T,
            param_dynamicImage
        )

        std_noise = estimate_std_over_time_optimized(processed_data, index_xmin, index_xmax)

        file_data_name = str(os.path.basename(output_data))
        with open(file_data_name, "w") as f:
            f.write(f"{mean_noise}\n{std_noise}\n")


        file_name = str(os.path.basename(output_image))
        # Suppression de l'extension .tif si présente
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

