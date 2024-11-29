try:
    library_loaded = False


    def load_library():
        global library_loaded
        if not library_loaded:
            import torch
            from pathlib import Path

            file_path = Path(__file__)
            so_files = list(file_path.parent.glob('*.so'))
            if not so_files:
                print('\033[1;41;37m No CUDA libraries found to load!\033[0m')
                return

            for f in so_files:
                torch.ops.load_library(f)

            library_loaded = True
            print('\033[1;42;38m CUDA libraries loaded successfully!\033[0m')
        else:
            print('\033[1;43;38m CUDA libraries already loaded!\033[0m')


    load_library()

except ImportError as e:
    print('\033[1;41;37m Failed to import torch! Make sure PyTorch is installed.\033[0m')
    print(e)

except OSError as e:
    print('\033[1;41;37m Failed to load CUDA libraries! Build the CUDA libraries first.\033[0m')
    print(e)

except Exception as e:
    print('\033[1;41;37m An unexpected error occurred!\033[0m')
    print(e)
