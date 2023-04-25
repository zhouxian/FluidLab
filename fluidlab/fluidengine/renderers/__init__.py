from fluidlab.utils.misc import is_on_server
try:
    from .ggui_renderer import GGUIRenderer
    print("Imported GGUIRenderer!")
except ImportError:
    print("Could not import GGUIRenderer")
try:
    from .gl_renderer import GLRenderer
    print("Imported GLRenderer!")
except ImportError:
    print("Could not import GLRenderer")