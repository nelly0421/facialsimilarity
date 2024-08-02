from django import template

register = template.Library()

@register.filter(name='remove_extension')
def remove_extension(value):
    """Remove the extension from a filename."""
    return value.rsplit('.', 1)[0]
