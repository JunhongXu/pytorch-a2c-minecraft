# # import gym
# # import gym_minecraft
import minecraft_py
# import logging
# # import MalmoPython
# import time
# import sys
# import numpy as np
# import cv2
# # ogger = logging.getLogger(__name__)
#
# # env = gym.make('MinecraftBasic-v0')
# # env.init(start_minecraft=True)
# # env.load_mission_file('envs/simple_navigation')
# # env.reset()
# #
# # done = False
# # while not done:
# #         env.render(mode='rgb_array')
# #         action = env.action_space.sample()
# #         obs, reward, done, info = env.step(action)
# #         print(obs.shape)
# # env.close()
#
# # def start():
# #     proc, port = minecraft_py.start()
# #     logger.log("Started Minecraft on port %d, overriding client_pool.", port)
# #     client_pool = [('127.0.0.1', port)]
# #     return MalmoPython.ClientPool()
#
# MAX_RETRY = 3
#
# # start minecraft game
# # minecraft_py.start()
#
# # create an agent
# agent = MalmoPython.AgentHost()
#
# # create the mission recorder
# mission_record_spec = MalmoPython.MissionRecordSpec()
#
# # load xml file to the mission
# with open('envs/simple_navigation.xml', 'r') as f:
#     mission = MalmoPython.MissionSpec(f.read(), True)
#
#
# # start the mission
# def init(agent, mission_record_spec):
#     for retry in range(MAX_RETRY):
#         try:
#             agent.startMission( mission, mission_record_spec )
#             break
#         except RuntimeError as e:
#             if retry == MAX_RETRY - 1:
#                 print "Error starting mission:",e
#                 exit(1)
#             else:
#                 time.sleep(2)
#
# for e in range(1):
#     init(agent, mission_record_spec)
#
#     print "Waiting for the mission to start ",
#     world_state = agent.getWorldState()
#     while not world_state.has_mission_begun:
#         sys.stdout.write(".")
#         time.sleep(0.1)
#         world_state = agent.getWorldState()
#         for error in world_state.errors:
#             print "Error:",error.text
#
#     while world_state.is_mission_running:
#         # sys.stdout.write(".")
#         time.sleep(0.1)
#         world_state = agent.getWorldState()
#         if len(world_state.video_frames):
#             frame = np.frombuffer(world_state.video_frames[0].pixels, np.uint8)
#             frame = np.reshape(frame, (240, 320, 3))
#             # print(frame.shape)
#             cv2.imwrite('test.png', frame)
#         # print(mission.requestVideo(300, 250))
#         # print(len(world_state.video_frames))
#         # print(world_state.number_of_observations_since_last_state, len(world_state.observations))
#         for error in world_state.errors:
#             print "Error:",error.text
