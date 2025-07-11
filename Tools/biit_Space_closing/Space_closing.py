import os
import sys
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
        dict(name='border_mode', help='Mode de gestion des bords (reflect, constant, etc.).', required=False, type='Str', default='reflect'),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='filledSpaceMorphology.tif', type='Path')
    ]


    def processAllData(self, argsList):
        """
        Traite toutes les données en appliquant la fermeture d'espace.

        Paramètres :
            argsList : liste d'objets avec les attributs nécessaires pour chaque image
        
        Retour :
            None
        """
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: __file__ location: {__file__}")
        print(f"DEBUG: sys.path: {sys.path[:3]}...")  # Afficher les 3 premiers éléments
        
        try:
            import numpy as np
            print("DEBUG: numpy imported successfully")
            from astroca.tools.loadData import load_data
            print("DEBUG: load_data imported successfully")
            from astroca.tools.exportData import export_data
            print("DEBUG: export_data imported successfully")
            from astroca.activeVoxels.spaceMorphology import closing_morphology_in_space
            print("DEBUG: closing_morphology_in_space imported successfully")
        except ImportError as e:
            print(f"DEBUG: First import failed: {e}")
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'astroca'))
            print(f"DEBUG: Trying to add to path: {base_dir}")
            print(f"DEBUG: Directory exists: {os.path.exists(base_dir)}")
            if os.path.exists(base_dir):
                print(f"DEBUG: Contents of {base_dir}: {os.listdir(base_dir)[:5]}...")  # Afficher les 5 premiers éléments
            
            if base_dir not in sys.path:
                sys.path.append(base_dir)
                print(f"DEBUG: Added {base_dir} to sys.path")
            else:
                print(f"DEBUG: {base_dir} already in sys.path")
                
            try:
                import numpy as np
                print("DEBUG: numpy imported successfully (second try)")
                from astroca.tools.loadData import load_data
                print("DEBUG: load_data imported successfully (second try)")
                from astroca.tools.exportData import export_data
                print("DEBUG: export_data imported successfully (second try)")
                from astroca.activeVoxels.spaceMorphology import closing_morphology_in_space
                print("DEBUG: closing_morphology_in_space imported successfully (second try)")
            except ImportError as e:
                print(f"DEBUG: Second import also failed: {e}")
                
                # Essayons une approche différente pour trouver astroca
                possible_paths = [
                    os.path.join(os.path.dirname(__file__), '..', '..', 'astroca'),
                    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'astroca'),
                    os.path.join(os.path.dirname(__file__), 'astroca'),
                    '/home/matteo/Bureau/INRIA/codePython/astroca',  # Chemin absolu basé sur votre structure
                ]
                
                for path in possible_paths:
                    abs_path = os.path.abspath(path)
                    print(f"DEBUG: Checking path: {abs_path}")
                    if os.path.exists(abs_path):
                        print(f"DEBUG: Found astroca at: {abs_path}")
                        if abs_path not in sys.path:
                            sys.path.append(abs_path)
                        try:
                            from astroca.tools.loadData import load_data
                            from astroca.tools.exportData import export_data
                            from astroca.dynamicImage.backgroundEstimator import background_estimation_single_block
                            print("DEBUG: Successfully imported astroca modules!")
                            break
                        except ImportError as e2:
                            print(f"DEBUG: Import failed even with path {abs_path}: {e2}")
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
        print(f"Shape of loaded data: {data.shape}")
        
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
            print(f"Data shape for time {t}: {data.shape}")
            data4D[t] = data

        print(f"Shape of merged data: {data4D.shape}")
        
        # Load xmin and xmax indices
        radius = int(argsList[0].radius)
        border_mode = str(argsList[0].border_mode)
        
        output_image = argsList[0].output_image

        # Apply the space closing operation
        processed_data = closing_morphology_in_space(data4D, radius, border_mode)

        # Save each time frame as a separate image
        file_name = str(os.path.basename(output_image))
        # remove .tif extension if present
        if file_name.endswith('.tif'):
            file_name = file_name[:-5]
        for t in range(time_length):
            file_name_t = f"{file_name}{t}.tif"
            data_to_export = processed_data[t][np.newaxis, ...]  # Add a new axis for time
            export_data(data_to_export, os.path.dirname(output_image), export_as_single_tif=True, file_name=file_name_t)
        
        
        
    
    