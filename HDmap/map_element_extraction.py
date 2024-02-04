import numpy as np

import pymap3d


from libs.ngiiParser import NGIIParser
from libs.map_elements_db import MapElementsDB

class MapElementExtractor:

    def __init__(self, cfg):
        self.cfg = cfg

        ngii = self._get_parsed_NGII_map(cfg.map_path)
        # self.base_lla = list(map(float, cfg.base_lla))
        self.base_lla = cfg.base_lla
        self.map_db = MapElementsDB(ngii, self.base_lla)

        self.precision = cfg.precision
        
    def _to_enu(self, lat, lon, alt):
        return pymap3d.geodetic2enu(lat, lon, alt, self.base_lla[0], self.base_lla[1], self.base_lla[2])


    def _get_parsed_NGII_map(self, map_path):

        ## Load map files
        a1_path = '%s/A1_NODE.shp'%(map_path)
        a2_path = '%s/A2_LINK.shp'%(map_path)
        a3_path = '%s/A3_DRIVEWAYSECTION.shp'%(map_path)
        a4_path = '%s/A4_SUBSIDIARYSECTION.shp'%(map_path)

        b1_path = '%s/B1_SAFETYSIGN.shp'%(map_path)
        b2_path = '%s/B2_SURFACELINEMARK.shp'%(map_path)
        b3_path = '%s/B3_SURFACEMARK.shp'%(map_path)

        c1_path = '%s/C1_TRAFFICLIGHT.shp'%(map_path)
        c3_path = '%s/C3_VEHICLEPROTECTIONSAFETY.shp'%(map_path)
        c4_path = '%s/C4_SPEEDBUMP.shp'%(map_path)
        c6_path = '%s/C6_POSTPOINT_Merge.shp'%(map_path)

        ngii = NGIIParser(
            a1_path,
            a2_path,
            a3_path,
            a4_path,
            b1_path, 
            b2_path, 
            b3_path, 
            c1_path,
            c3_path,
            c4_path,
            c6_path)

        return ngii


    def _extract_points(self, target_element : str, linkID_of_interest : list):
        points_dict = {}

        for current_linkID in linkID_of_interest:

            for id_, coords in self.map_db.link_dict[current_linkID][target_element].items():

                length = len(coords)
                points = np.zeros([3, length])
 
                for i, (lon, lat, alt) in enumerate(coords):
                    points[:, i]  = np.array(self._to_enu(lat, lon, alt))

                points_dict[id_] = points

        return points_dict


    def _interpolate_points(self, points):
        pass

    def _get_id_of_interest(self, current_linkID : str, depth=2, is_only_my_lane=False):
        
        axis1 = [current_linkID]

        axis2 = []
        ## Next1
        for next1_linkID in self.map_db.link_dict[current_linkID]['NEXT']:
            axis1.append(next1_linkID)

            if depth>=2:
                ## Next2
                for next2_linkID in self.map_db.link_dict[next1_linkID]['NEXT']:
                    axis1.append(next2_linkID)
                
                    if depth>=3:
                        ## Next3
                        for next3_linkID in self.map_db.link_dict[next2_linkID]['NEXT']:
                            axis1.append(next3_linkID)
                        
                            if depth >= 4:
                                ## Next4
                                for next4_linkID in self.map_db.link_dict[next3_linkID]['NEXT']:
                                    axis1.append(next4_linkID)
                                    
                                    if depth >= 5:
                                        ## Next5
                                        for next5_linkID in self.map_db.link_dict[next4_linkID]['NEXT']:
                                            axis1.append(next5_linkID)



        if is_only_my_lane:
            ## Something's wrong...
            for linkID in axis1:
                ## Left1
                for left1_linkID in self.map_db.link_dict[linkID]['LEFT']:
                    axis2.append(left1_linkID)
                ## Right1
                for right1_linkID in self.map_db.link_dict[linkID]['RIGHT']:
                    axis2.append(right1_linkID)


        else:
            ## Prev1
            for prev1_linkID in self.map_db.link_dict[current_linkID]['PREV']:
                axis1.append(prev1_linkID)

                if depth>=2:
                    ## prev2
                    for prev2_linkID in self.map_db.link_dict[prev1_linkID]['PREV']:
                        axis1.append(prev2_linkID)


            for linkID in axis1:
                ## Left1
                for left1_linkID in self.map_db.link_dict[linkID]['LEFT']:
                    axis2.append(left1_linkID)


                    if depth>=2:
                        ## Left2
                        for left2_linkID in self.map_db.link_dict[left1_linkID]['LEFT']:
                            axis2.append(left2_linkID)

                            if depth >=3:
                                ## Left3
                                for left3_linkID in self.map_db.link_dict[left2_linkID]['LEFT']:
                                    axis2.append(left3_linkID)

                                    if depth >=4:
                                        ## Left4
                                        for left4_linkID in self.map_db.link_dict[left3_linkID]['LEFT']:
                                            axis2.append(left4_linkID)



                ## Right1
                for right1_linkID in self.map_db.link_dict[linkID]['RIGHT']:
                    axis2.append(right1_linkID)


                    if depth>=2:
                        ## right2
                        for right2_linkID in self.map_db.link_dict[right1_linkID]['RIGHT']:
                            axis2.append(right2_linkID)





        return [*axis1, *axis2]






    def _generate_straight_lane(self, x, isVertical=True):
        length = 20
        n = 50
        if isVertical:
            xs = np.ones(n) * x
            ys = np.linspace(0, 10, n)
        else: #is Horizontal
            xs = np.linspace(0, 10, n)
            ys = np.ones(n) * x
 
        zs = np.ones(n) * 0
        ones = np.ones(n)

        elem = np.hstack([xs, ys, zs, ones]).reshape(4, n)
        return elem

    def get_element_by_id(self, id_:str):
        coords = self.map_db.get_element_by_id(id_)

        length = len(coords)
        points = np.zeros([3, length])

        for i, (lon, lat, alt) in enumerate(coords):
            points[:, i]  = np.array(self._to_enu(lat, lon, alt))
        return np.vstack([points, np.ones(length)])



    def run(self, element_names, vehicle_pose_WC, isSeperated=False, is_only_my_lane=False, depth=2, target_ids=[]):
        
        if self.map_db.isQueryInitialized:
            current_linkID = self.map_db.get_current_linkID(vehicle_pose_WC)
        else: # not initialized, not use previous linkID
            current_linkID = self.map_db.initialize_query(vehicle_pose_WC)

        elems_WC_dict = {}


        linkID_of_interest = self._get_id_of_interest(current_linkID, is_only_my_lane=is_only_my_lane, depth=depth)

        for target_element in element_names:
            
            if target_element == "HELP":
                print("Available elements: LINEMARK, TRAFFICLIGHT")

            if target_element in ["LINEMARK", "SAFETYSIGN", "SURFACEMARK", "TRAFFICLIGHT"]:
                points_dict = self._extract_points(target_element, linkID_of_interest)

                tmp_points = None

                for points in points_dict.values():
                    if not isSeperated:
                        tmp_points = np.hstack([tmp_points, points]) if tmp_points is not None else points
                    else:
                        tmp_points = tmp_points + [points] if tmp_points is not None else [points]

                if tmp_points is not None:
                    if not isSeperated:
                        elems_WC_dict[target_element] = np.vstack([tmp_points, np.ones(tmp_points.shape[1])])
                    else:
                        elems_WC_dict[target_element] = tmp_points

        #end for
        return elems_WC_dict

# if __name__ == "__main__":
    # ext = MapElementExtractor(cfg)
    # ext.run()

    # vehicle_pose = np.zeros(6) # x, y, z, roll, pitch, yaw
    # elems_WC = ext.run(["SAFETYSIGN"], vehicle_pose, isSeperated=True, depth=depth)


