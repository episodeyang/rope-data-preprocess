import lmdb
import caffe
import matplotlib.pyplot as plt
import time


class Config:
    transpose = True
    img_size = 0


# use the folder, not the data file as the path to the lmdb
DB = lmdb.open('test/image_after')
with DB.begin(write=False) as txn:
    i = 0
    cursor = txn.cursor()
    for key, img_datum in cursor:
        i += 1
        print("reading key:{key}".format(key=key))
        datum = caffe.proto.caffe_pb2.Datum()
        datum.ParseFromString(img_datum)
        img = caffe.io.datum_to_array(datum)  # .astype(np.uint8)
        if not Config.img_size:
            print(img.shape)
            img_size = img.shape
        if Config.transpose:
            img = img.transpose()
        if i <= 3:
            plt.imshow(img, cmap='gray')
            plt.show()
            time.sleep(1.0)
        if i == 3:
            break
            exit()
# DB = lmdb.open('test/image_after')
# with DB.begin(write=True) as txn:
#     for key, img_datum in txn.cursor():
#         print("reading key:{key}".format(key=key))
#         datum = caffe.proto.caffe_pb2.Datum()
#         datum.ParseFromString(img_datum)
#         img = caffe.io.datum_to_array(datum)  # .astype(np.uint8)
#         if not Config.img_size:
#             print(img.shape)
#             img_size = img.shape
#         if Config.transpose:
#             img = img.transpose()
#         plt.imshow(img, cmap='gray')
#         plt.show()
#         exit()
#         # cv2.waitKey(millis)
