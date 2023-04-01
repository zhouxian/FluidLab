from fluidlab.utils.misc import is_on_server
if not is_on_server():
    from .ggui_renderer import GGUIRenderer
    from .gl_renderer import GLRenderer