import os
import sys
import subprocess

class Tool():
    
    name = "Background estimator"
    description = "This tool estimates the background of a 4D image sequence (T,Z,Y,X) using a moving window approach."
    categories = ['Astroca', 'Florescence Estimation']
    environment = 'astroca-env'

    dependencies = dict(
        python='==3.10',
        conda=['tqdm', 'numpy', 'pandas', 'numba'],
        pip=[]
    )

    inputs = [
        dict(name='input_image', help='Chemin vers le fichier .tif 4D (T,Z,Y,X).', required=True, type='Path', autoColumn=True),
        dict(name='index_xmin', help='Chemin vers le fichier .npy contenant les xmin par Z.', required=True, type='Path'),
        dict(name='index_xmax', help='Chemin vers le fichier .npy contenant les xmax par Z.', required=True, type='Path'),
        dict(name='moving_window', help="Window size for background estimation.", required=False, type='Int', default=7),
    ]

    outputs = [
        dict(name='output_image', help='The output image.', 
             default='F0_estimated.tif', type='Path')
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
        Process the input data to estimate the background of a 4D image sequence.

        Parameters:
            args : object with attributes for input_image, index_xmin, index_xmax, and moving_window.
        """

        # Setup the environment
        self.setup_environment()

        # Import necessary modules after environment setup
        try:
            import numpy as np
            from astroca.tools.loadData import load_data
            from astroca.tools.exportData import export_data
            from astroca.dynamicImage.backgroundEstimator import background_estimation_single_block_numba as background_estimation
        except ImportError as e:
            raise ImportError("Required modules could not be imported. " 
                            "Ensure that the 'astroca' package is installed correctly.") from e

        first_volume = str(args.input_image)

        # Check if the input file exists
        if not os.path.exists(first_volume):
            raise FileNotFoundError(f"Input file not found: {first_volume}")

        data = load_data(first_volume)
        if data.ndim == 3:
            data = data[np.newaxis, ...]

        index_xmin_path = str(args.index_xmin)
        index_xmax_path = str(args.index_xmax)
        # Check if index files exist
        if not os.path.exists(index_xmin_path):
            raise FileNotFoundError(f"Index xmin file not found: {index_xmin_path}")
        if not os.path.exists(index_xmax_path):
            raise FileNotFoundError(f"Index xmax file not found: {index_xmax_path}")
        index_xmin = np.load(index_xmin_path)
        index_xmax = np.load(index_xmax_path)

        moving_window = int(args.moving_window)
        output_image = str(args.output_image)

        param_background_estimation = {
            'background_estimation': {
                'moving_window': moving_window,
                'method': 'percentile',
                'method2': 'Med',
                'percentile': 10,
            },
            'save': {'save_background_estimation': 0},
            'paths': {'output_dir': None}
        }

        processed_data = background_estimation(
            data,
            index_xmin,
            index_xmax,
            param_background_estimation
        )

        # Save the processed image
        file_name = str(os.path.basename(output_image))
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