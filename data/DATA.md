# Dead Sea Scrolls Training Data

The pipeline automatically downloads all training data.
Ensure that the `.env` file is present in the project's root directory to make downloading from password protected sources possible.
We use a train-dev-test split of 80%-10%-10% for all data.
An explanation of the train-dev-test split can be found on <https://cs230.stanford.edu/blog/split/>.
The data loader expects the following structure of the data.

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
                
        font-characters/
            training/
                ...
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
