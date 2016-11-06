import numpy as np
import sys
from PotetialGeneration.SavableObject import ThisComputer
import Pyro4
import os
import sdl2.ext
import time

Pyro4.config.SERIALIZER = 'pickle'

__author__ = 'anton'

#os.environ["PYSDL2_DLL_PATH"] = "/home/lithium/GitHub/DMD/pyDMD/TIDLPcontrol/VideoDisplayer"

def test():
    RESOURCES = sdl2.ext.Resources(__file__, "resources")
    sdl2.ext.init()

    window = sdl2.ext.Window("Hello World!", size=(100, 100),flags=sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP)
    window.show()
    window.maximize()
    #sdl2.SDL_

    factory = sdl2.ext.SpriteFactory(sdl2.ext.SOFTWARE)
    sprite = factory.from_image(RESOURCES.get_path("out.png"))

    spriterenderer = factory.create_sprite_render_system(window)
    spriterenderer.render(sprite)
    processor = sdl2.ext.TestEventProcessor()
    processor.run(window)

video_displayer_settings = {'default_image': None}

class SoftwareRenderer(sdl2.ext.SoftwareSpriteRenderSystem):
    def __init__(self, window):
        super(SoftwareRenderer, self).__init__(window)

    def render(self, components):
        sdl2.ext.fill(self.surface, sdl2.ext.Color(0, 0, 0))
        super(SoftwareRenderer, self).render(components)

class DmdImage(sdl2.ext.Entity):

    def __init__(self, world, sprite):
        self.sprite = sprite
        self.world = world


class DmdServerPoller(object):

    def __init__(self):
        self.comp = ThisComputer()
        name = "PYRONAME:LiLab.dmdserver.%s"%self.comp.hostname
        self.dmd_server = Pyro4.Proxy(name)
        self.dmd_server.ping()

        self.current_uuid = None
        self.filepath = None


    def check_and_download(self):
        if self.current_uuid is None or self.dmd_server.check_for_updates(self.current_uuid):
            print self.dmd_server.download_img()
            self.filepath, self.current_uuid = self.dmd_server.download_img()
            self.dmd_server.clean_temp_directory()
            return True, self.filepath
        else:
            return False, self.filepath


def run():
    RESOURCES = sdl2.ext.Resources(__file__, "resources")
    sdl2.ext.init()
    window = sdl2.ext.Window("DMD DISPLAY", size = (1920, 1080), flags=sdl2.SDL_WINDOW_FULLSCREEN_DESKTOP)
    server_poller = DmdServerPoller()
    window.show()
    window.maximize()

    world = sdl2.ext.World()

    factory = sdl2.ext.SpriteFactory(sdl2.ext.SOFTWARE)

    is_new, filepath = server_poller.check_and_download()
    dmd_sprite = factory.from_image(RESOURCES.get_path(filepath))

    spriterenderer = SoftwareRenderer(window)
    world.add_system(spriterenderer)
    dmd_image = DmdImage(world, dmd_sprite)

    running = True

    while running:
        events = sdl2.ext.get_events()
        is_new, filepath = server_poller.check_and_download()

        if is_new:
            sdl2.SDL_Delay(10)
            print filepath
            print "is_valid %s"%os.path.isfile(os.path.join('resources',filepath))
            dmd_image.delete()
            RESOURCES.add_file(os.path.join('resources',filepath))
            #RESOURCES = sdl2.ext.Resources(__file__, "resources")
            dmd_sprite = factory.from_image(RESOURCES.get_path(filepath))
            dmd_image = DmdImage(world, dmd_sprite)
            print "updating image to %s" % filepath

        for event in events:
            if event.type == sdl2.SDL_QUIT:
                running = False
                break
        window.refresh()
        world.process()
        sdl2.SDL_Delay(100)
    return 0

if __name__ == "__main__":
    #test()
    sys.exit(run())
