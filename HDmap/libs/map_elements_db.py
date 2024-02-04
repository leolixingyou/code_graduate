import numpy as np

import pymap3d

from libs.ngiiParser import NGIIParser

from scipy.spatial import cKDTree

class MapElementsDB:
    def __init__(self, ngii : NGIIParser, base_lla):
        self.base_lla = base_lla
        self.link_KDT_metadata = {}
        self.link_refpoints = []

        nodeIDs = self._get_nodeIDs(ngii.a1_node)
        self.link_start_from = self._set_node_dict(nodeIDs)
        self.link_end_at = self._set_node_dict(nodeIDs)
        self.link_dict = self._set_link_dict(ngii.a2_link)
        self.link_KDTree = self._get_KDTree(self.link_refpoints)
    
        for nodeID in nodeIDs:

            for linkID in self.link_start_from[nodeID]:
                self.link_dict[linkID]['PREV'] = \
                                self.link_end_at[nodeID]

            for linkID in self.link_end_at[nodeID]:
                self.link_dict[linkID]['NEXT'] = \
                                self.link_start_from[nodeID]

        self._parse_elements_to_link("SAFETYSIGN", ngii.b1_safetysign)
        self._parse_elements_to_link("LINEMARK", ngii.b2_surfacelinemark)
        self._parse_elements_to_link("SURFACEMARK", ngii.b3_surfacemark)
        self._parse_elements_to_link("TRAFFICLIGHT", ngii.c1_trafficlight)

        ###################CCC
        self._parse_elements_to_link("POSTPOINT", ngii.c6_postpoint)
        ######################



        self.isQueryInitialized = False
        self.prev_linkID = None

    def get_element_by_id(self, ngii: NGIIParser, element_id: str):
        for attr_name in vars(ngii):
            element_list = getattr(ngii, attr_name)
            for element in element_list:
                if element.ID == element_id:
                    return element.geometry.coords
        raise ValueError(f"No element found with id {element_id}")

    def initialize_query(self, vehicle_pose_WC):
        x = vehicle_pose_WC[0:2]
        dists, idxs = self.link_KDTree.query(x, k=10, p=2, workers=1)

        print(f"\n===============\nNearest Link Query from {x}\n")
        for i, idx in enumerate(idxs):
            print(f"Rank#{i}   idx: {idx}  dist: {dists[i]} link: {self.link_KDT_metadata[str(idx)]}")
        print("\n===================")

        current_linkID = self.link_KDT_metadata[str(idxs[0])]
        self.isQueryInitialized = True
        self.prev_linkID = current_linkID
        return current_linkID

    def get_current_linkID(self, vehicle_pose_WC):

        ##TODO: find current linkID using prev link ID and current pose

        current_linkID = self.prev_linkID
        self.prev_linkID = current_linkID
        return current_linkID



    def _get_nodeIDs(self, a1_node):
        nodeIDs = []
        for node in a1_node:
            nodeIDs.append(node.ID)
        return nodeIDs
 
    def _set_node_dict(self, nodeIDs : list):
        node_dict = {}
        for nodeID in nodeIDs:
            node_dict[nodeID] = []
        return node_dict


    def _set_link_dict(self, a2_link):
        link_dict = {}
        for link in a2_link:

            ## Init link dict
            link_dict[link.ID] = self._get_standard_dict()

            ## Add Adjacent LinkIDs
            if type(link.R_LinkID) is str:
                link_dict[link.ID]['RIGHT'].append(link.R_LinkID)

            if type(link.L_LinKID) is str:
                link_dict[link.ID]['LEFT'].append(link.L_LinKID)

            ## Add Next & Prev NodeIDs
            self.link_start_from[link.FromNodeID].append(link.ID)
            self.link_end_at[link.ToNodeID].append(link.ID)

            ## Add Reference Points (lon, lat, alt)
            kdt_idx0 = len(self.link_KDT_metadata.keys())

            ### Add First Coordinate (lon, lat, alt)
            self.link_refpoints.append(link.geometry.coords[0])
            self.link_KDT_metadata[str(kdt_idx0)] = link.ID

            ### Add Middle Coordinate (lon, lat, alt)
            mid_idx = len(link.geometry.coords) // 2
            self.link_refpoints.append(link.geometry.coords[mid_idx])
            self.link_KDT_metadata[str(kdt_idx0+1)] = link.ID
 
            ### Add Last Coordinate (lon, lat, alt)
            self.link_refpoints.append(link.geometry.coords[-1])
            self.link_KDT_metadata[str(kdt_idx0+2)] = link.ID

        return link_dict

    def _get_standard_dict(self):
        return {'NEXT':[], 'PREV':[], 'LEFT': [], 'RIGHT': [], 
                'MIDPOINT': ('lon', 'lat', 'alt'), 
                'LINEMARK': {}, 'TRAFFICLIGHT': {},
                'SAFETYSIGN' : {}, 'SURFACEMARK' : {}}

    def _get_KDTree(self, points:list):
        tmp = np.array(points)        
        tmp = pymap3d.geodetic2enu(tmp[:, 1], tmp[:, 0], tmp[:, 2], self.base_lla[0], self.base_lla[1], self.base_lla[2])
        tmp = np.array(tmp)[0:2, :] ## eliminate altitude
        return cKDTree(tmp.T)

    def _parse_elements_to_link(self, target_name : str, ngii_elements : list):
        for element in ngii_elements:

            if target_name == "LINEMARK":
                if type(element.L_linkID) is str: #not NaN
                    self.link_dict[element.L_linkID][target_name][element.ID] = element.geometry.coords

                elif type(element.R_linkID) is str:
                    self.link_dict[element.R_linkID][target_name][element.ID] = element.geometry.coords

            ## TODO: Implement other elements
            elif target_name == "SAFETYSIGN":
                if type(element.LinkID) is str: #not NaN
                    self.link_dict[element.LinkID][target_name][element.ID] = element.geometry.boundary.coords

            elif target_name == "TRAFFICLIGHT":
                if type(element.LinkID) is str: #not NaN
                    self.link_dict[element.LinkID][target_name][element.ID] = element.geometry.coords

            elif target_name == "SURFACEMARK":
                if type(element.LinkID) is str: #not NaN
                    self.link_dict[element.LinkID][target_name][element.ID] = element.geometry.boundary.coords
