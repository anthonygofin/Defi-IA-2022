kaggle competitions download -c defi-ia-2022
wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/test/X_forecast/2D_arpege_test.tar.gz
wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2016/2D_arpege_2016.tar.gz
wget https://meteonet.umr-cnrm.fr/dataset/data/defi_ia_challenge/train/X_forecast/2017/2D_arpege_2017.tar.gz

mkdir DATA_RAINFALL
echo "Extracting X_station (train & test)......."
unzip  defi-ia-2022.zip -d DATA_RAINFALL
echo "Extracting X_forecast (test)......."
tar -xf 2D_arpege_test.tar.gz
echo "Extracting X_forecast (train, 2016)......."
tar -xf 2D_arpege_2016.tar.gz
echo "Extracting X_forecast (train, 2017)......."
tar -xf 2D_arpege_2017.tar.gz

echo "Moving files at the right place......."
mkdir ./DATA_RAINFALL/Train/Train/2D_arpege_train
mv ./2D_arpege_2016 ./DATA_RAINFALL/Train/Train/2D_arpege_train/2D_arpege_2016
mv ./2D_arpege_2017 ./DATA_RAINFALL/Train/Train/2D_arpege_train/2D_arpege_2017

mv ./2D_arpege ./DATA_RAINFALL/Test/Test/2D_arpege_test
echo "Done ! :D"


