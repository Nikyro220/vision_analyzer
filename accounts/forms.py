from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm

from .models import User


class RegisterForm(UserCreationForm):
    email = forms.EmailField(
        required=False,
        label="Email",
        widget=forms.EmailInput(attrs={"placeholder": "you@example.com", "autocomplete": "email"}),
    )

    class Meta:
        model = User
        fields = ("username", "email", "password1", "password2")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update(
            {"placeholder": "Придумайте логин", "autocomplete": "username", "autofocus": True}
        )
        self.fields["password1"].widget.attrs.update(
            {"placeholder": "Придумайте пароль", "autocomplete": "new-password"}
        )
        self.fields["password2"].widget.attrs.update(
            {"placeholder": "Повторите пароль", "autocomplete": "new-password"}
        )
        for f in self.fields.values():
            f.help_text = None

    def clean_username(self):
        username = self.cleaned_data["username"]
        if User.objects.filter(username__iexact=username).exists():
            raise forms.ValidationError("Пользователь с таким логином уже существует.")
        return username


class LoginForm(AuthenticationForm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["username"].widget.attrs.update(
            {"placeholder": "Логин", "autocomplete": "username", "autofocus": True}
        )
        self.fields["password"].widget.attrs.update(
            {"placeholder": "Пароль", "autocomplete": "current-password"}
        )

    error_messages = {
        "invalid_login": "Неверный логин или пароль.",
        "inactive": "Этот аккаунт деактивирован.",
    }


class ProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ("email", "first_name", "last_name")
        widgets = {
            "email": forms.EmailInput(attrs={"placeholder": "you@example.com"}),
            "first_name": forms.TextInput(attrs={"placeholder": "Имя"}),
            "last_name": forms.TextInput(attrs={"placeholder": "Фамилия"}),
        }
        labels = {
            "email": "Email",
            "first_name": "Имя",
            "last_name": "Фамилия",
        }
