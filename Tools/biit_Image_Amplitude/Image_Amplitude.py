import os
import sys
import subprocess

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
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', required=True, type='Path', autocolumn=True),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', required=True, type='Path', autoColumn=True),
    ]

    outputs = [
        dict(name='output_image', help='The output image.', 
             default='inverse_anscombe_transformed_volume.tif', type='Path')
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
        Traite les données en appliquant l'estimation de l'amplitude avec la transformation inverse d'Anscombe.
        Paramètres :
            args : objet avec les attributs nécessaires pour l'image d'entrée, l'image de fond, et les indices xmin/xmax

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
            from astroca.dynamicImage.dynamicImage import compute_image_amplitude

        except ImportError as e:
            raise ImportError("Impossible d'importer les modules nécessaires. "
                              "Vérifiez que le module 'astroca' est présent.") from e

        raw_data_path = str(args.input_image)
        # Vérification du fichier d'entrée
        if not os.path.exists(raw_data_path):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {raw_data_path}")
        raw_data = load_data(raw_data_path)

        f0_path = str(args.f0_image)
        if not os.path.exists(f0_path):
            raise FileNotFoundError(f"Le fichier d'image de fond est introuvable : {f0_path}")
        f0_data = load_data(f0_path)
        if f0_data.ndim == 3:
            f0_data = f0_data[np.newaxis, ...]

        index_xmin_path = str(args.index_xmin)
        index_xmax_path = str(args.index_xmax)
        # Vérification des fichiers d'indices
        if not os.path.exists(index_xmin_path):
            raise FileNotFoundError(f"Le fichier index_xmin est introuvable : {index_xmin_path}")
        if not os.path.exists(index_xmax_path):
            raise FileNotFoundError(f"Le fichier index_xmax est introuvable : {index_xmax_path}")
        index_xmin = np.load(index_xmin_path)
        index_xmax = np.load(index_xmax_path)

        params_amplitude = {
            'save': {'save_amplitude': 0,
                     'save_anscombe_inverse': 0
                     },
            'paths': {'output_dir': None},
        }

        output_image = args.output_image

        processed_data = compute_image_amplitude(raw_data, f0_data, index_xmin, index_xmax, params_amplitude)

        file_name = str(output_image)
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


