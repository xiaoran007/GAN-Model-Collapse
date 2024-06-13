from GAN import get_default_device
import DatasetsLoader
from plots import GANTsne, SMOTETsne, CTGANTsne, Imgs


if __name__ == "__main__":
    # for i in DatasetsLoader.Datasets_list:
    for i in ['SouthGermanCredit']:
        print("Start training for dataset: ", i)

        gan_obj = GANTsne(dataset_name=i, device=get_default_device(force_skip_mps=False))
        gan_obj.fit()
        gan_obj.draw_and_save(epoch=10)
        gan_obj.draw_and_save(epoch=50)
        gan_obj.draw_and_save(epoch=100)
        gan_obj.draw_and_save(epoch=149)

        smote_obj = SMOTETsne(dataset_name=i)
        smote_obj.draw_and_save()

        ctgan_obj = CTGANTsne(dataset_name=i)
        ctgan_obj.fit()
        ctgan_obj.draw_and_save()

        imgs_obj = Imgs(dataset_name=i)
        imgs_obj.draw_and_save()

