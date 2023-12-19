# ML-Repository
Repository for ML-Learning Path

## How To Use The Model
### First Load the Model and Scaler(Pickle)
```python
import tensorflow as tf
import pickle
from keras import load_model

model_path = "your_model_path/model_name.tflite"
tflite_interpreter = tf.lite.Interpret(model_path=model_path)
tflite_interpreter.allocate_tensors()

with open('scaler_path/name_of_the_scaler.pkl', 'rb') as file:
    loaded_scaler = pickle.load(file)
```

### Prepare the New Data Point
```python
new_data_point = np.array([-7.87639, 110.35889, 1, 0, 1, 1, 1])
#This is example of the data point, consisting of [latitude,longitude,kategori1,kategori2,kategori3,kategori4,kategori5]

#We transform the longitude and latitude with Scaler
new_data_point[:2] = loaded_scaler.transform(new_data_point[:2].reshape(1,-1))

#Reshape it before predicting
new_data_point = new_data_point.reshape(1, -1)

#Set the input tensor
tflite_interpreter.set_tensor(tflite_interpreter.get_input_details()[0]['index'], new_data_point)
```
### Making Prediction
```python
tflite_interpreter.invoke()
predictions = tflite_interpreter.get_tensor(tflite_interpreter.get_output_details()[0]['index'])

#Get the probabilities of class predictions
class_probabilities = predictions[0]

#Sorting it
sorted_indices = np.argsort(class_probabilities1)[::-1]
next_most_likely_indices = sorted_indices1[1:16] #Here it return 15 places id, you can change it accordingly

#Inside the next_most_likely_indices
[607 608 644 623 645 613 664 636 621 651 649 618 652 655 640]
#You can search this id in the database to get data of laundry places
```

### If you call the rows by its index from next_most_likely_indices, you will receive this result
| id  | nama                                            | no_telp     | alamat                                                                                                                                                             | latitude           | longitude          | kategori_1 | kategori_2 | kategori_3 | kategori_4 | kategori_5 |
|-----|-------------------------------------------------|-------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------|--------------------|------------|------------|------------|------------|------------|
| 607 | Laundry Rungkut Surabaya "Rumah Laundry"        | 81234567891 | Jl. Pejaten Barat Raya No.40A, RT.2/RW.8, West Pejaten, Pasar Minggu, South Jakarta City, Jakarta 12540                                                            | 0.0026057142939869 | 0.6757323160620077 | 0          | 0          | 0          | 1          | 0          |
| 608 | Ryns Laundry                                    | 81234567891 | Jl. Raya Pulo Gebang No.34, RT.14/RW.4, Ujung Menteng, Kec. Cakung, Kota Jakarta Timur, Daerah Khusus Ibukota Jakarta 13960                                        | 0.0031159449663874 | 0.6748606404075828 | 1          | 1          | 1          | 1          | 1          |
| 644 | GRESS LAUNDRY SURABAYA                          | 81234567891 | Jl. Gading Raya No.703, RT.5/RW.8, Pd. Bambu, Kec. Duren Sawit, Kota Jakarta Timur, Daerah Khusus Ibukota Jakarta 13430                                            | 0.0                | 0.6746492918535596 | 0          | 0          | 0          | 0          | 1          |
| 623 | LAUNDRY TIME                                    | 81234567891 | Jl. Perserikatan No.4 A- B, RT.2/RW.8, Rawamangun, Kec. Pulo Gadung, Kota Jakarta Timur, Daerah Khusus Ibukota Jakarta 13220                                       | 0.0034871945583115 | 0.6735064070187748 | 1          | 1          | 1          | 1          | 1          |
| 645 | LAUNDRY ALL IN ONE-UR ONE STOP LAUNDRY SERVICES | 81234567891 | Jl. Haji Nurisan No 22 Rt 02 / Rw 011 No 17E, RT.2/RW.11, Pd. Pinang, Kec. Kby. Lama, Jakarta, Daerah Khusus Ibukota Jakarta 12310                                 | 0.0039098381956201 | 0.6714211108252952 | 1          | 1          | 1          | 1          | 1          |
| 613 | Natura Laundry & Dry Clean Nginden              | 81234567891 | Jl. Kartini IV Dalam No.128 C, Kartini, Sawah Besar, Central Jakarta City, Jakarta 10750                                                                           | 0.0045298241093085 | 0.6750065954008813 | 0          | 1          | 1          | 1          | 1          |
| 664 | Laundry 35                                      | 81234567891 | Jl. Kemang Raya No.130d, RT.3/RW.2, Bangka, Kec. Mampang Prpt., Kota Jakarta Selatan, Daerah Khusus Ibukota Jakarta 12730                                          | 0.0042194692225673 | 0.6746813447148341 | 0          | 1          | 0          | 0          | 0          |
| 636 | Sa'iki londri                                   | 81234567891 | Signature Park Grande, Apartement, Jl. Letjen M.T. Haryono No.kav 20, RT.4/RW.1, Cawang, Kec. Kramat jati, Kota Jakarta Timur, Daerah Khusus Ibukota Jakarta 13630 | 0.0044110206205939 | 0.673278650452434  | 0          | 1          | 1          | 1          | 0          |
| 621 | IPSO Laundromat                                 | 81234567891 | Jl. Kp. Utan No.36, RT.15/RW.5, Ragunan, Pasar Minggu, South Jakarta City, Jakarta 12550                                                                           | 0.0063644467575785 | 0.672815362592984  | 1          | 0          | 1          | 1          | 1          |
| 651 | The Daily Wash Nginden                          | 81234567891 | Jl. Pembangunan III No.2B, RT.13/RW.1, Petojo Utara, Kecamatan Gambir, Kota Jakarta Pusat, Daerah Khusus Ibukota Jakarta 10130                                     | 0.0049464959033154 | 0.6755113802682304 | 1          | 1          | 1          | 1          | 1          |
| 649 | Professional Laundry Surabaya - Marckalaundry   | 81234567891 | Perumahan Daan Mogot baru jl Gilimanuk blok JD no 43 A, RT.8/RW.12, Kalideres, Kec. Kalideres, Kota Jakarta Barat, Daerah Khusus Ibukota Jakarta 11480             | 0.0050805909301814 | 0.6749173529850445 | 1          | 1          | 1          | 1          | 1          |
| 618 | Netto Laundromat Wiyung                         | 81234567891 | Jl. Gatot Subroto No.300, Menteng Dalam, Tebet, South Jakarta City, Jakarta 12780                                                                                  | 0.0037330354408993 | 0.6717502727235827 | 0          | 0          | 1          | 0          | 0          |
| 652 | TheDailyWash Ngagel Surabaya                    | 81234567891 | Jl. Bekasi Timur Raya No.5A, RT.5/RW.8, Cipinang, Kec. Pulo Gadung, Kota Jakarta Timur, Daerah Khusus Ibukota Jakarta 13240                                        | 0.0055656674747354 | 0.6747693946879156 | 1          | 1          | 1          | 1          | 1          |
| 655 | CLEAN LAUNDRY                                   | 81234567891 | Gedung Mitra Hadiprana, Parking Entry, Jl. Kemang Raya Jl. Kemang IV No.30, Bangka, Mampang Prapatan, South Jakarta City, Jakarta 12730                            | 0.0063163100812676 | 0.6715987847958971 | 0          | 1          | 1          | 1          | 1          |
| 640 | Laundry Time                                    | 81234567891 | Jl. Mandar III Gg. Musholah No.3, Pondok Karya, Pondok Aren, South Tangerang City, Banten 15225                                                                    | 0.0064552006792699 | 0.6735588745000269 | 0          | 0          | 0          | 1          | 0          |
