from django import template
import builtins

register = template.Library()

@register.filter
def lookup(dictionary, key):
    """Template filter to lookup dictionary values by key"""
    if isinstance(dictionary, dict):
        return dictionary.get(key, '')
    return ''

@register.filter
def get_item(dictionary, key):
    """Alternative template filter to get dictionary items"""
    return dictionary.get(key)

@register.filter
def mul(value, arg):
    """Multiply the value by the argument"""
    try:
        return float(value) * float(arg)
    except (ValueError, TypeError):
        return 0

@register.filter
def abs(value):
    """Return absolute value"""
    try:
        return builtins.abs(float(value))
    except (ValueError, TypeError):
        return 0