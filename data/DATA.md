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
                    *.pgm 
                .../
                    *.pgm   # 300*0.8  images for each Hebrew character in pgm format
                Zayin/
                    *.pgm
            dev/
                Alef/
                    *.pgm 
                .../
                    *.pgm   # 300*0.1 images for each Hebrew character in pgm format
                Zayin/
                    *.pgm
            test/
                Alef/
                    *.pgm 
                .../
                    *.pgm   # 300*0.1 images for each Hebrew character in pgm format
                Zayin/
                    *.pgm
                
        font_characters/
            training/
                Alef.jpeg
                ...
                Zayin.jpeg

            Habbakuk.TTF

        fragments/
            train/
                ...
            dev/
                ...
            test/
                ...

        character_styles/
            ...

        fragment_styles/
            ...
