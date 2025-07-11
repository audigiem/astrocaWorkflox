import os
import sys
class Tool():
    
    # The display name
    name = "Boundaries Computation"
    # The tool description is important for the user to understand what the tool does
    description = "This tool computes the boundaries of 3D+time sequence."
    # The category which defines where the tool will apear in the tool library (the Tools tab)
    categories = ['Astroca', 'Boundaries']
    # The name of the conda environment which will be used to run the tool
    # It will be created when needed unless the BioImageIT environment does not satisfy the requirements 
    # (in this case the tool we be run in the BioImageIT environment)
    # You can set it to BioImageIT if you are sure the tool only requires packages installed with BioImageIT 
    environment = 'astroca-env'
    # The tool dependencies:
    # - the python version
    # - the conda packages which will be installed with 'conda install packageName'
    # - the pip packages which will be installed with 'pip install packageName'
    dependencies = dict(python='==3.10', conda=['tqdm', 'skimage'], pip=[])
    # The inputs
    inputs = [
        dict(name='input_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path', autoColumn=True),
        dict(name='x_min', help='Minimum x coordinate for cropping', required=True, type='Int'),
        dict(name='x_max', help='Maximum x coordinate for cropping', required=True, type='Int'),
        dict(name='pixel_cropped', help='Number of pixels to crop from the height dimension.', required=True, type='Int'),
    ]

    outputs = [
        dict(name='output_image', help='Image transformée sauvegardée.', default='data_cropped.tif', type='Path'),
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', default='index_xmin.npy', required=True, type='Path'),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', default='index_xmax.npy', required=True, type='Path')
    ]
    
    
    def processAllData(self, argsList):
        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data, save_numpy_tab
            from astroca.croppingBoundaries.computeBoundaries import compute_boundaries
            from astroca.croppingBoundaries.cropper import crop_boundaries
        except ImportError as e:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'astroca'))
            if base_dir not in sys.path:
                sys.path.append(base_dir)
            try:
                import numpy as np
                from astroca.tools.loadData import load_data
                from astroca.croppingBoundaries.computeBoundaries import compute_boundaries
                from astroca.tools.exportData import export_data, save_numpy_tab
                from astroca.croppingBoundaries.cropper import crop_boundaries
            except ImportError as e:
                raise ImportError("Impossible d'importer les modules nécessaires. "
                                "Vérifiez que le module 'astroca' est présent.") from e
                
        time_length = len(argsList)
        first_volume = argsList[0].input_image
        first_volume = str(first_volume)  # Ensure it's a string path
        # Vérification du fichier d'entrée
        if not os.path.exists(first_volume):
            raise FileNotFoundError(f"Le fichier d'entrée est introuvable : {first_volume}")
        # Chargement des données
        data = load_data(first_volume)
        Z, Y, X = data.shape
        data4D = np.empty((time_length, Z, Y, X), dtype=data.dtype)
        data4D[0] = data  # Initialize the first time frame
        # Merge all the others input data into one 4D array
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
            
        print(f"Shape of merged data: {data4D.shape}")

        x_min = argsList[0].x_min
        x_max = argsList[0].x_max
        pixel_cropped = argsList[0].pixel_cropped
        output_image = argsList[0].output_image
                
        
        params = {
            'preprocessing': {
                'x_min': x_min,
                'x_max': x_max,
                'pixel_cropped': pixel_cropped
            },
            'files': {'save_results': 0},
            'paths': {'output_dir': None}
        }
        index_xmin, index_xmax, _, processed_data = compute_boundaries( crop_boundaries(data4D, params), params)
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
            
        save_numpy_tab(index_xmin, os.path.dirname(output_image), file_name="index_xmin.npy")
        save_numpy_tab(index_xmax, os.path.dirname(output_image), file_name="index_xmax.npy")

        

        
        

    

