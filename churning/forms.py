from django import forms


class CustomerForm(forms.Form):
    customer_file = forms.FileField()
