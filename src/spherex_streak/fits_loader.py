from astropy.io import fits
import numpy as np
import os


def load_fits_image(file_path):
    
    if not os.path.exists(file_path):
        raise FileExistsError(f"الملف غير موجود في المسار: {file_path}")
    with fits.open(file_path) as hdul:
        hdul.info()
        
        data = hdul[0].data.astype(np.float32) if hdul[0].data is not None else hdul[1].data.astype(np.float32)
        header = hdul[0].header
        
    return data , header

if __name__ == "__main__":
    sample_path = "data/raw.sample.fits"
    print(جاري محاولة قراءة الملف التجريبي من: {sample_path})
    
    try:
        data, header = load_fits_image(sample_path)
        print(f"تم قراءة الصورة بنجاح!")
        print(f"أبعاد مصفوفة البيانات (Shape): {data.shape}")
        print(f"نوع بيانات المصفوفة: {data.dtype}")
    except Exception as e:
        print(f"ملاحظة: {e}")
        print("يرجى وضع ملف FITS تجريبي داخل مجلد data/raw/ للبدء في الاختبار الفعلي.")