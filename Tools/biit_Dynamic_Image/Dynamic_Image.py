import os
import sys
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
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', required=True, type='Path'),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', required=True, type='Path'),
        dict(name='time_length', help='Longueur temporelle de la séquence.', required=False, type='Int', default=1),
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
            from astroca.dynamicImage.dynamicImage import compute_dynamic_image
        except ImportError as e:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'astroca'))
            if base_dir not in sys.path:
                sys.path.append(base_dir)
            try:
                import numpy as np
                from astroca.tools.loadData import load_data
                from astroca.tools.exportData import export_data
                from astroca.dynamicImage.dynamicImage import compute_dynamic_image
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
        
        # Load xmin and xmax indices
        F0 = argsList[0].background_image
        F0 = str(F0)  # Ensure it's a string path
        # Vérification du fichier d'entrée
        if not os.path.exists(F0):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {F0}")
        dataF0 = load_data(F0)
        
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
        
        # Ensure the time_length matches the number of time frames in data4D
        if time_length != data4D.shape[0]:
            raise ValueError(f"La longueur temporelle spécifiée ({time_length}) ne correspond pas au nombre de trames temporelles dans les données ({data4D.shape[0]}).")
        
        
        output_image = argsList[0].output_image
        if dataF0.ndim == 3:
            # If dataF0 is 3D, we need to expand it to 4D by adding a new axis for time
            dataF0 = dataF0[np.newaxis, ...]
        
        if dataF0.ndim != 4:
            raise ValueError(f"Le fichier de fond doit être un tableau 4D (nbF0, Z, Y, X), mais a une forme {dataF0.shape}.")
        
        param_dynamicImage = {
            'files': {'save_results': 0},
            'paths': {'output_dir': None}
        }
        
        # Apply the Anscombe variance stabilization
        processed_data, mean_noise = compute_dynamic_image(
            data4D,
            dataF0,
            index_xmin,
            index_xmax,
            time_length,
            param_dynamicImage
        )
        
        # print(f"Processed data shape: {processed_data.shape}")
        
        # Save each time frame as a separate image
        file_name = str(os.path.basename(output_image))
        # remove .tif extension if present
        if file_name.endswith('.tif'):
            file_name = file_name[:-5]
        for t in range(time_length):
            file_name_t = f"{file_name}{t}.tif"
            data_to_export = processed_data[t][np.newaxis, ...]  # Add a new axis for time
            # print(f"Export Data shape: {data_to_export.shape}, File Name: {file_name_t}")
            export_data(data_to_export, os.path.dirname(output_image), export_as_single_tif=True, file_name=file_name_t)
        
        
        
    
    