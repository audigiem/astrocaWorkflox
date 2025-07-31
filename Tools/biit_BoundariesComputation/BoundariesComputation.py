import os
import sys
import subprocess


class Tool():
    
    # The display name
    name = "Boundaries Computation"
    # The tool description is important for the user to understand what the tool does
    description = "This tool computes the boundaries of 3D+time sequence."
    # The category which defines where the tool will apear in the tool library (the Tools tab)
    categories = ['Astroca', 'Boundaries']
    # Environnement personnalisé qui sera créé
    environment = 'astroca-env'
    # Dependencies de base + votre repo GitHub
    dependencies = dict(
        python='==3.10',
        conda=['tqdm', 'scikit-image', 'git', 'poetry'], 
        pip=[],
    )
    
    # Les inputs restent identiques
    inputs = [
        dict(name='input_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path', autoColumn=True),
        dict(name='x_min', help='Minimum x coordinate for cropping', required=True, type='Int', default=0),
        dict(name='x_max', help='Maximum x coordinate for cropping', required=True, type='Int', default=319),
        dict(name='pixel_cropped', help='Number of pixels to crop from the height dimension.', required=True, type='Int', default=10),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='data_cropped.tif', type='Path'),
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', default='index_xmin.npy', required=True, type='Path'),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', default='index_xmax.npy', required=True, type='Path')
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
        # Configuration de l'environnement
        self.setup_environment()
        
        # Import des modules après installation
        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data, save_numpy_tab
            from astroca.croppingBoundaries.computeBoundaries import compute_boundaries
            from astroca.croppingBoundaries.cropper import crop_boundaries
        except ImportError as e:
            raise ImportError(f"Impossible d'importer les modules astroca: {e}")
                
        first_volume = str(args.input_image)
        
        # Vérification du fichier d'entrée
        if not os.path.exists(first_volume):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {first_volume}")
            
        # Chargement des données
        data = load_data(first_volume)

        x_min = args.x_min
        x_max = args.x_max
        pixel_cropped = args.pixel_cropped
        output_image = args.output_image
                
        params = {
            'preprocessing': {
                'x_min': x_min,
                'x_max': x_max,
                'pixel_cropped': pixel_cropped
            },
            'save': {
                'save_cropp_boundaries': 0,
                'save_boundaries': 0
            },
            'paths': {'output_dir': None}
        }
        
        index_xmin, index_xmax, _, processed_data = compute_boundaries(
            crop_boundaries(data, params), params
        )
        
        # Save results
        file_name = str(os.path.basename(output_image))
        if file_name.endswith('.tif'):
            file_name = file_name[:-4]

        export_data(processed_data, os.path.dirname(output_image), 
                   export_as_single_tif=True, file_name=file_name)
        save_numpy_tab(index_xmin, os.path.dirname(output_image), 
                      file_name="index_xmin.npy")
        save_numpy_tab(index_xmax, os.path.dirname(output_image), 
                      file_name="index_xmax.npy")


    def processAllData(self, argsList):
        for args in argsList:
            try:
                self.processData(args)
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {args.input_image}: {e}")
                continue

