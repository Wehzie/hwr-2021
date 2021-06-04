# Dead Sea Scrolls Training Data

The pipeline automatically downloads all training data.
Ensure that the `.env` file is present in the project's root directory to make downloading from password protected sources possible.
We use a train-dev-test split of 80%-10%-10% for all data.

To load all data run the following command.

    python3 src/data_handler/dataset_builder.py

The data loader produces the following data structuring.

    data/
        characters/
            train/
                Alef/
                    *.jpg
                .../
                    *.jpg   # 300*0.8  images for each Hebrew character in jpg format
                Zayin/
                    *.jpg
            dev/
                Alef/
                    *.jpg 
                .../
                    *.jpg   # 300*0.1 images for each Hebrew character in jpg format
                Zayin/
                    *.jpg
            test/
                Alef/
                    *.jpg 
                .../
                    *.jpg   # 300*0.1 images for each Hebrew character in jpg format
                Zayin/
                    *.jpg
                
        font_characters/
            training/
                Alef/
                    *.jpeg 
                .../
                    *.jpeg   # 30 morphed font character images for pretraining
                Zayin/
                    *.jpeg

            Habbakuk.TTF

        fragments/
            train/
                ...
            dev/
                ...
            test/
                ...
        
        segmented_lines/
            train/
                fragment_1/         
                    line_1.pgm      
                    ...
                    line_n.pgm      # a .pgm for each line in a fragment
                ...
                document_n/         # a directory for each fragment
            dev/
                ...
            test/                   # train, dev, test split following the fragments
                ...

        segmented_characters/
            ...

        character_styles/
            ...

        fragment_styles/
            ...

        model/                      # trained tensorflow models
            custom/
                ...
            dense_net_121/
                ...