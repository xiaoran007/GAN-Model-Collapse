from GAN import get_default_device
import DatasetsLoader
from plots import GANTsne, SMOTETsne, CTGANTsne, Imgs, TsneInit, AllTsne, AllTsneWithMajority


if __name__ == "__main__":
    for i in DatasetsLoader.Datasets_list:
    # for i in ['BankNote']:
        print("Start training for dataset: ", i)

        all_tsne_obj = AllTsneWithMajority(dataset_name=i, device=get_default_device(force_skip_mps=False))
        all_tsne_obj.fit()

        # init_obj = TsneInit(dataset_name=i)
        # init_obj.fit()
        # init_real = init_obj.getTsneInitReal()
        #
        # gan_obj = GANTsne(dataset_name=i, device=get_default_device(force_skip_mps=False))
        # gan_obj.fit()
        # gan_obj.draw_and_save(epoch=9, init_real=init_real)
        # gan_obj.draw_and_save(epoch=49, init_real=init_real)
        # gan_obj.draw_and_save(epoch=99, init_real=init_real)
        # gan_obj.draw_and_save(epoch=149, init_real=init_real)
        #
        # smote_obj = SMOTETsne(dataset_name=i)
        # smote_obj.draw_and_save(init_real=init_real)
        #
        # ctgan_obj = CTGANTsne(dataset_name=i)
        # ctgan_obj.fit()
        # ctgan_obj.draw_and_save(init_real=init_real)
        #
        imgs_obj = Imgs(dataset_name=i)
        imgs_obj.draw_and_save()

