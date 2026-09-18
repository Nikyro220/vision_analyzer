from django import forms


class ImageUploadForm(forms.Form):
    image = forms.ImageField(
        label="Изображение",
        widget=forms.ClearableFileInput(attrs={"accept": "image/*"}),
    )
