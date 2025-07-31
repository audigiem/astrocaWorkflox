import os
import sys
import subprocess
class Tool():
    # Nom affiché dans BioImageIT
    name = "Space Closing"
    
    # Description visible pour l'utilisateur
    description = "Applies a space closing operation on a 4D image sequence (T,Z,Y,X) to enhance the features by closing small gaps in the data."
    
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
        dict(name='input_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path', autoColumn=True),
        dict(name='radius', help='Rayon pour l\'opération de fermeture.', required=True, type='Int', default=1),
        dict(name='border_mode', help='Mode de gestion des bords (reflect, constant, etc.).', required=False, type='Str', default='ignore'),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='filledSpaceMorphology.tif', type='Path')
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
        Traite les données en appliquant la fermeture d'espace.

        Paramètres :
            args : objet avec les attributs nécessaires pour l'image d'entrée, le rayon et le mode de bordure

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
            from astroca.activeVoxels.spaceMorphology import closing_morphology_in_space
        except ImportError as e:
            raise ImportError("Impossible d'importer les modules nécessaires. "
                              "Vérifiez que le module 'astroca' est présent.") from e

        first_volume = str(args.input_image)
        # Vérification du fichier d'entrée
        if not os.path.exists(first_volume):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {first_volume}")
        data = load_data(first_volume)

        radius = int(args.radius)
        border_mode = str(args.border_mode)
        output_image = args.output_image

        processed_data = closing_morphology_in_space(data, radius, border_mode)

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
