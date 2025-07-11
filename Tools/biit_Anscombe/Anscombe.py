import os
import sys
class Tool():
    # Nom affiché dans BioImageIT
    name = "Anscombe Variance Stabilization"
    
    # Description visible pour l'utilisateur
    description = "Stabilise la variance d'une séquence 4D (T,Z,Y,X) avec la transformation d'Anscombe."
    
    # Catégorie dans laquelle l'outil apparaîtra
    categories = ['Astroca', 'Variance Stabilization']

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
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', required=True, type='Path'),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', required=True, type='Path'),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='{input_image.stem}_variance_stabilized.tif', type='Path')
    ]
 
        
    
    def processAllData(self, argsList):
        """
        Traite toutes les données en appliquant la variance stabilisation.
        
        Paramètres :
            argsList : liste d'objets avec les attributs nécessaires pour chaque image
        
        Retour :
            None
        """
        # Import avec fallback si lancement local
        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data
            from astroca.varianceStabilization.varianceStabilization import compute_variance_stabilization
        except ImportError as e:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'astroca'))
            if base_dir not in sys.path:
                sys.path.append(base_dir)
            try:
                import numpy as np
                from astroca.tools.loadData import load_data
                from astroca.tools.exportData import export_data
                from astroca.varianceStabilization.varianceStabilization import compute_variance_stabilization
            except ImportError as e:
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
        
        output_image = argsList[0].output_image
        
        param_anscombe = {
            'files': {'save_results': 0},
            'paths': {'output_dir': None}
        }
        
        # Apply the Anscombe variance stabilization
        processed_data = compute_variance_stabilization(
            data4D,
            index_xmin,
            index_xmax,
            param_anscombe
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
        
        
        
    
    