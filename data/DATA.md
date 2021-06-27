# Dead Sea Scrolls Training Data

The pipeline automatically downloads all training data.
Ensure that the `.env` file is present in the project's root directory to make downloading from password protected sources possible.
We use a train-dev-test split of 80%-10%-10% for all labelled data.
We plan to use a dev-test split of 50%-50% for unlabelled fragment data.

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
            fragment1.jpg
            ...
        
        segmented_fragments/
            dev/
                ...
            test/
                ...
        
        character_styles/
            Archaic/
                Alef/
                    *.jpg 
                .../
                    *.jpg
                Zayin/
                    *.jpg
            Hasmonean/
                Alef/
                    *.jpg 
                .../
                    *.jpg
                Zayin/
                    *.jpg
            Herodian
                Alef/
                    *.jpg 
                .../
                    *.jpg
                Zayin/
                    *.jpg


        fragment_styles/
            full_image_periods/
                Archaic/
                    fragment1.jpg
                    ...
                Hasmonean
                    fragment1.jpg
                    ...
                Herodian
                    fragment1.jpg
                    ...

        model/                      # trained tensorflow models
            recognizer/
                model_0/
                    assets/
                    variables/
                    saved_model.pb
                ...
            style/
                model_0/
                    assets/
                    variables/
                    saved_model.pb
                ...
            
        work_dir/                   # this folder name is reserved by the pipeline
            ---empty---             # as working directory
                                    # files inside this folder will be automatically deleted
        
        out/                        # we recommend this folder name as output directory
            ---empty---             # for the pipeline

Applying character segmentation to a single fragment image results in the following output.
Character segmentation populates the `segmented_fragments` directory.

    fragment_name/
        fragment_name.jpg
        lines/
            line_L0.png
            ...
            line_Ln.png
        words/
            word_L0_W0.png
            ...
            word_Ln_Ln.png
        characters/
            character_L0_W0_C0.png
            ...
            character_Ln_Wn_Cn.png
