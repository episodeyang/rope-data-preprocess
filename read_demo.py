import os
# from tqdm import tqdm

import lmdb
import caffe
import cv2
# import matplotlib.pyplot as plt
# import time


class Config:
    transpose = True
    img_size = 0


# use the folder, not the data file as the path to the lmdb


def get_image(path):
    DB = lmdb.open(path)
    with DB.begin(write=False) as txn:
        cursor = txn.cursor()
        # keys = [key for key, datum in cursor]
        # print(keys[:5])
        # print(len(keys))
        # print(*sorted(keys))
        # label = ('{:08}'.format(start)).encode('ascii')
        # cursor.set_key(label)
        # datum = cursor.get(label)
        for i, (key, datum) in enumerate(cursor):
            cv_datum = caffe.proto.caffe_pb2.Datum()
            cv_datum.ParseFromString(datum)
            img = caffe.io.datum_to_array(cv_datum)
            if Config.transpose:
                img = img.transpose()
            yield key, img
            # plt.imshow(img, cmap='gray')
            # plt.show()
            # time.sleep(1.0)
            # print("plotting: ", key)
            # if i == 3:
            #     break

        # i = 0
        # for key, datum in cursor:
        #     i += 1
        #     print("reading key:{key}".format(key=key))
        #     cv_datum = caffe.proto.caffe_pb2.Datum()
        #     cv_datum.ParseFromString(datum)
        #     img = caffe.io.datum_to_array(cv_datum)
        #     if Config.transpose:
        #         img = img.transpose()
        #     plt.imshow(img, cmap='gray')
        #     plt.show()
        #     time.sleep(1.0)
        #     if i >= 3:
        #         exit()


def get_pos(path):
    DB = lmdb.open(path)
    with DB.begin(write=False) as txn:
        i = 0
        cursor = txn.cursor()
        # cursor.next()
        keys = [key for key, datum in cursor]
        print('==========')
        print(len(keys))
        print(*sorted(keys))
        # for key, img_datum in cursor:
        #     i += 1
        #     print("reading key:{key}".format(key=key))
        #     datum = caffe.proto.caffe_pb2.Datum()
        #     datum.ParseFromString(img_datum)
        #     vec = caffe.io.datum_to_array(datum)  # .astype(np.uint8)
        #     exit()


if __name__ == "__main__":
    print('this is going to take a while (mostly the training set), be patient.')
    for res in [(64, 64), (240, 240)]:
        res_str = "{}x{}".format(*res)
        for which in ["test", "train"]:
            output_dir = "../causal-infogan-rope/{}/{}".format(res_str, which)
            print(output_dir)
            os.makedirs(output_dir + "/before", exist_ok=True)
            os.makedirs(output_dir + "/after", exist_ok=True)
            for (k1, img_before), (k2, img_after) in zip(get_image(which + '/image_before'),
                                                         get_image(which + '/image_after')):
                resized_1 = cv2.resize(img_before, res, interpolation=cv2.INTER_AREA)
                resized_2 = cv2.resize(img_after, res, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(output_dir, "before", "{}.jpg".format(k1.decode())), resized_1)
                cv2.imwrite(os.path.join(output_dir, "after", "{}.jpg".format(k2.decode())), resized_2)
            print('finished', output_dir)
        print('finished', res_str)
    print('finished!')
