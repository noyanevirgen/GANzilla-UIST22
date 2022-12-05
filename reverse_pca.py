from flask import Flask, render_template, redirect, url_for,request
from flask import make_response
from flask_cors import CORS, cross_origin
import yaml
app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
import legacy
import torch
import dnnlib
device = torch.device('cuda')
with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl') as f:
#with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl') as f:
#with dnnlib.util.open_url('https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/afhqcat.pkl') as f:
    image_size = 1024
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
import stylegan2 as sty
import dnnlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision.utils import save_image
import os
import clip
from torchvision.datasets import CIFAR100
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import json
from collections import defaultdict
import pickle
import random
import time
from flask import jsonify
import glob
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
args = dnnlib.EasyDict()
args.G_kwargs = dnnlib.EasyDict(class_name='training.networks.Generator', z_dim=512, w_dim=512, mapping_kwargs=dnnlib.EasyDict(), synthesis_kwargs=dnnlib.EasyDict())
args.G_kwargs.synthesis_kwargs.channel_base = 32768
args.G_kwargs.synthesis_kwargs.channel_max = 512
args.G_kwargs.synthesis_kwargs.num_fp16_res = 4 # enable mixed-precision training
args.G_kwargs.synthesis_kwargs.conv_clamp = 256 # clamp activations to avoid float16 overflow

affines = ['affine0', 'affine1']
weights = ['weight', 'bias']


@app.route('/login', methods=['GET', 'POST'])
def login():
    #This is the first scatter, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    
    
    if request.method == 'POST':
        datafromjs = request.form['mydata']
        app.logger.info('overhereee')
        app.logger.info(type(datafromjs))
        all_seeds = yaml.load(datafromjs)
        app.logger.info(all_seeds)
        all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
        all_w = OG.mapping(torch.from_numpy(all_z).to(device), None)
        styles = OG.styling(all_w)
        all_images = OG.synthesis(all_w, styles, noise_mode='const') #found images and filters
        real_selected_filters = []
        n_examples = 30
        chosen_img = 0
        for image_idx in range(5):
            I = np.asarray(Image.open('public/ganzilla_images/initial_images/highlight_eyes' + str(image_idx) + '.png'))
            mask = np.mean(I[:,:,:3],2) == 0
            mask = torch.from_numpy(mask).to(dtype=torch.float32).cuda()
            selected_filters = []
            threshold = 0.5
            increase = 0.12
            for i in range(layers_number):
                if i == 0:
                    conv1 = all_images[1][i][1][0]
                    conv1 = F.interpolate(conv1[:,None,:,:], size=(image_size,image_size))
                    conv1 = torch.squeeze(conv1)
                    overlap1 = torch.mean(mask * torch.abs(conv1), axis=(1,2))
                    overlap1 = overlap1 / torch.mean(mask)
                    selected_filters.append((None,~(overlap1 > threshold)))
                else:
                    conv0 = all_images[1][i][0][0]
                    conv0 = F.interpolate(conv0[:,None,:,:], size=(image_size,image_size))
                    conv0 = torch.squeeze(conv0)
                    overlap0 = torch.mean(mask * torch.abs(conv0), axis=(1,2))
                    overlap0 = overlap0 / torch.mean(mask)
                    conv1 = (all_images[1][i][1][0])
                    conv1 = F.interpolate(conv1[:,None,:,:], size=(image_size,image_size))
                    conv1 = torch.squeeze(conv1)
                    overlap1 = torch.mean(mask * torch.abs(conv1), axis=(1,2))
                    overlap1 = overlap1 / torch.mean(mask)
                    selected_filters.append((~(overlap0 > threshold), ~(overlap1 > threshold)))
            real_selected_filters.append(selected_filters)
        selected_filters = real_selected_filters[0]

        for s_filter in real_selected_filters:
            for layer_idx, layer in enumerate(s_filter):
                if layer_idx == 0:
                    selected_filters[layer_idx] = (None,selected_filters[layer_idx][1] * layer[1])
                else:
                    selected_filters[layer_idx] = ((selected_filters[layer_idx][0] * layer[0], selected_filters[layer_idx][1] * layer[1]))
                    
        for i in range(len(selected_filters)):
            if i == 0:
                selected_filters[i] = (None, ~(selected_filters[i][1] > 0)) 
            else:
                selected_filters[i] = (~(selected_filters[i][0]>0), ~(selected_filters[i][1]>0))         
        for idx, i in enumerate(selected_filters):
            if idx == 0:
                app.logger.info(i[1].sum())
            else:
                app.logger.info(i[0].sum())
                app.logger.info(i[1].sum())
        #filters are selected
        #create initial 30 images
        directions = []
        

        new_styles = OG.styling(all_w)
        new_style = []
        #make it faster by just calculating index 0 in the future

        for i in range(n_examples):
            direction_of_image = []
            new_styles = OG.styling(all_w)
            new_style = []
            for idx, selects in enumerate(selected_filters):
                if idx == 0:
                    sel_styles = styles[idx+1][0][:,selects[1]] #select 0,1 -> style 1,0
                    #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                    dir1 = sel_styles + (torch.randint(2, size = sel_styles.shape)*torch.rand(size = sel_styles.shape)-0.5).cuda()*10*increase
                    new_styles[idx+1][0][:,selects[1]] = dir1
                    direction_of_image.append((None, dir1))
                elif idx <= layers_number-2:
                    sel_styles = styles[idx][1][:,selects[0]] # select 1,0 style 1,1
                    #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                    if idx == 2 or idx == 3:
                        dir0 = sel_styles + (torch.randint(2, size = sel_styles.shape)*torch.rand(size = sel_styles.shape)-0.5).cuda()*30*increase
                    else:
                        dir0 = sel_styles + (torch.randint(2, size = sel_styles.shape)*torch.rand(size = sel_styles.shape)-0.5).cuda()*10*increase
                    new_styles[idx][1][:,selects[0]] = dir0
                    sel_styles = styles[idx+1][0][:,selects[1]] #select 1,1 style 2,0
                    #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                    if idx == 2 or idx == 3:
                        dir1 = sel_styles + (torch.randint(2, size = sel_styles.shape)*torch.rand(size = sel_styles.shape)-0.5).cuda()*30*increase
                    else:
                        dir1 = sel_styles + (torch.randint(2, size = sel_styles.shape)*torch.rand(size = sel_styles.shape)-0.5).cuda()*10*increase
                    new_styles[idx+1][0][:,selects[1]] = dir1
                    direction_of_image.append((dir0, dir1))
                else:
                    sel_styles = styles[idx][1][:,selects[0]] # select 8,0 style 8,1
                    #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                    dir0 = sel_styles + (torch.randint(2, size = sel_styles.shape)*torch.rand(size = sel_styles.shape)-0.5).cuda()*10*increase
                    new_styles[idx][1][:,selects[0]] = dir0
                    direction_of_image.append((dir0, None))
            directions.append(direction_of_image)#make this a dictionary in the future
            new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
            new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
            save_image(new_images_tmp[chosen_img], 'public/ganzilla_images/raw_images/' + str(all_seeds[0]) + '/img' + str(i) +'.png')

        #Cluster these 30 images
        img_names = []
        img_featues = []
        for idx, image in enumerate(range(n_examples)): #fix this, so it works for everything and create a function
            img_p = 'public/ganzilla_images/raw_images/' + str(all_seeds[0]) + '/img' + str(image)+'.png'
            app.logger.info(img_p)
            if ('.png') in img_p:
                with Image.open(img_p) as im:
                    image_input = preprocess(im).unsqueeze(0).to(device)
                    text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
            

                # Calculate features
                with torch.no_grad():
                    image_features = model.encode_image(image_input)
                    #text_features = model.encode_text(text_inputs)

                image_features /= image_features.norm(dim=-1, keepdim=True)
                img_featues.append(image_features)
                img_names.append(img_p)
        
        #Should be a function
        chosen_img = 0 #should not be used in the future when direction is only calculated for first image
        directions_list_tmp = []
        for sample_id,direction in enumerate(directions):
            dir_tmp = direction[0][1][chosen_img]
            for i in range(1,layers_number-1):
                dir_tmp = torch.cat((dir_tmp,direction[i][0][chosen_img],direction[i][1][chosen_img]))
            dir_tmp = torch.cat((dir_tmp,direction[layers_number-1][0][chosen_img]))
            directions_list_tmp.append(dir_tmp.unsqueeze(0))
        directions_list = torch.cat(directions_list_tmp)
        directions_list = directions_list.cpu().detach().numpy()
        np.save('directions_list.npy',np.array(directions_list))
        pickle.dump(directions, open( "directions.pkl", "wb" ) )
        pickle.dump(selected_filters, open ('selected_filters.pkl', 'wb'))
        #clustering
        img_features = torch.cat(img_featues)
        img_tmp = np.array(img_names)
        n_clusters = 6
        kmeans = KMeans(n_clusters=n_clusters).fit(img_features.cpu())
        labels = kmeans.labels_

        average_embeddings = np.zeros((n_clusters,directions_list.shape[1]))
        direction_per_cluster = np.zeros((n_clusters,directions_list.shape[1]))
        for label in range(n_clusters):
            chosen_direction_per_cluster = directions_list[np.where(labels==label)]
            average_embeddings[label] = np.mean(chosen_direction_per_cluster,0) #average embedding for a cluster
        for selected_label in range(n_clusters):
            chosen_cluster_average_direction = average_embeddings[selected_label]
            cur_direction = 0
            for label in range(n_clusters):
                if label != selected_label:
                    cur_cluster_average_direction = average_embeddings[label]
                    tmp_direction_current = chosen_cluster_average_direction - cur_cluster_average_direction
                    tmp_direction_current = tmp_direction_current/np.linalg.norm(tmp_direction_current)
                    cur_direction += tmp_direction_current
            cur_direction = cur_direction/np.linalg.norm(cur_direction)
            direction_per_cluster[selected_label] = cur_direction

        locations = TSNE(n_components=2,init='random').fit_transform(direction_per_cluster)
        np.save('locations.npy',np.array(locations,dtype=int))
        np.save('labels.npy', labels)
        resp = defaultdict(list)
        for idx, i in enumerate(range(6)):
            representative_img = np.where(labels==i)[0]
            list_of_destinations = []
            for img_number in representative_img:
                list_of_destinations.append('ganzilla_images/raw_images/' + str(all_seeds[0]) + '/img' + str(img_number) +'.png')
            resp[i] = [int(locations[idx][0]), int(locations[idx][1]), list_of_destinations]        
        #locations = locations.tolist()
        resp = json.dumps(resp)
        resp = make_response(resp)
        return resp
@app.route('/testpjs', methods=['GET', 'POST'])
def test():
    n_examples= 30
    selected = request.form['mydata']
    selected = json.loads(selected)
    chosen_img = 0 #should not be used in the future when direction is only calculated for first image
    all_seeds = selected[2]
    number_of_clicks = selected[1]
    selected = selected[0]
    all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    all_w = OG.mapping(torch.from_numpy(all_z).to(device), None)
    new_styles = OG.styling(all_w)
    with open('directions_list.npy', 'rb') as f:
        directions_list = np.load(f)
    with open('labels.npy', 'rb') as f:
        labels = np.load(f)
    directions = pickle.load( open( 'directions.pkl', "rb" ) )
    selected_filters = pickle.load( open( 'selected_filters.pkl', "rb" ) )
    direction_subset = []
    for select_id,selects in enumerate(selected):
        if selects:
            app.logger.info(select_id)
            direction_indices = np.where(labels == select_id)
            for cur in directions_list[direction_indices]:
                direction_subset.append(cur)
    direction_subset = np.array(direction_subset) 
    direction = directions[0]



    
    
    increase = 0.3
    directions_list_new = np.zeros((n_examples,directions_list.shape[1]))
    for cur_example_id in range(n_examples):
        dir_tmp = []
        chsn = direction[0][1][chosen_img].shape[0]
        pc_values = random.choice(direction_subset)
        pc_values2 = random.choice(direction_subset)
        pc_values = (pc_values+pc_values2)/2
        app.logger.info('hereeeeeeeeeeeeeeeeeeee')
        pc_values += (np.random.rand(pc_values.shape[0])-0.5)*10*increase*np.random.randint(2, size = pc_values.shape[0])
        
        directions_list_new[cur_example_id] = pc_values
        dir_tmp.append((None,torch.tensor(pc_values[:chsn]).float()))
        for i in range(1,layers_number-1):
            chsn0 = direction[i][0][chosen_img].shape[0]
            chsn1 = direction[i][1][chosen_img].shape[0]
            dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn0]).float(),torch.tensor(pc_values[chsn+chsn0:chsn+chsn0+chsn1]).float()))
            chsn += chsn0 + chsn1
        chsn1 = direction[layers_number-1][0][chosen_img].shape[0]
        dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn1]).float(),None))

        new_styles = OG.styling(all_w)
        for idx, selects in enumerate(selected_filters):
            if idx == 0:
                dir1 = dir_tmp[0][1]
                new_styles[idx+1][0][chosen_img][selects[1]] = torch.clone(dir1.cuda()) 

            elif idx <= layers_number-2:
                if idx == 2 or idx == 3:
                    dir0 = dir_tmp[idx][0]
                else:
                    dir0 = dir_tmp[idx][0]
                new_styles[idx][1][chosen_img][selects[0]] = torch.clone(dir0.cuda())

                if idx == 2 or idx == 3:
                    dir1 = dir_tmp[idx][1]
                else:
                    dir1 = dir_tmp[idx][1]
                new_styles[idx+1][0][chosen_img][selects[1]] = torch.clone(dir1.cuda())


            else:
                dir0 = dir_tmp[idx][0]
                new_styles[idx][1][chosen_img][selects[0]] = torch.clone(dir0.cuda())   
        new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
        new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
        pathh='public/ganzilla_images/test/' + str(all_seeds[0]) + '_' + str(number_of_clicks-1)
        if not os.path.exists(pathh):
            os.mkdir(pathh)
        save_image(new_images_tmp[chosen_img], 'public/ganzilla_images/test/' + str(all_seeds[0]) + '_' + str(number_of_clicks-1) + '/img' + str(cur_example_id) +'.png')
    
    #Cluster these 30 images
    img_names = []
    img_featues = []
    for idx, image in enumerate(range(n_examples)): #fix this, so it works for everything and create a function
        img_p = 'public/ganzilla_images/test/' + str(all_seeds[0]) + '_' + str(number_of_clicks-1) + '/img' + str(image)+'.png'
        app.logger.info(img_p)
        if ('.png') in img_p:
            with Image.open(img_p) as im:
                image_input = preprocess(im).unsqueeze(0).to(device)
                text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
        

            # Calculate features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                #text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_featues.append(image_features)
            img_names.append(img_p)
    
    #Should be a function
    
    directions_list_tmp = []
    for sample_id,direction in enumerate(directions):
        dir_tmp = direction[0][1][chosen_img]
        for i in range(1,layers_number-1):
            dir_tmp = torch.cat((dir_tmp,direction[i][0][chosen_img],direction[i][1][chosen_img]))
        dir_tmp = torch.cat((dir_tmp,direction[layers_number-1][0][chosen_img]))
        directions_list_tmp.append(dir_tmp.unsqueeze(0))
    directions_list = torch.cat(directions_list_tmp)
    directions_list = directions_list.cpu().detach().numpy()
    np.save('directions_list.npy',np.array(directions_list_new))
    #clustering
    img_features = torch.cat(img_featues)
    img_tmp = np.array(img_names)
    n_clusters = 6
    kmeans = KMeans(n_clusters=n_clusters).fit(img_features.cpu())
    labels = kmeans.labels_

    average_embeddings = np.zeros((n_clusters,directions_list.shape[1]))
    direction_per_cluster = np.zeros((n_clusters,directions_list.shape[1]))
    for label in range(n_clusters):
        chosen_direction_per_cluster = directions_list[np.where(labels==label)]
        average_embeddings[label] = np.mean(chosen_direction_per_cluster,0) #average embedding for a cluster
    for selected_label in range(n_clusters):
        chosen_cluster_average_direction = average_embeddings[selected_label]
        cur_direction = 0
        for label in range(n_clusters):
            if label != selected_label:
                cur_cluster_average_direction = average_embeddings[label]
                tmp_direction_current = chosen_cluster_average_direction - cur_cluster_average_direction
                tmp_direction_current = tmp_direction_current/np.linalg.norm(tmp_direction_current)
                cur_direction += tmp_direction_current
        cur_direction = cur_direction/np.linalg.norm(cur_direction)
        direction_per_cluster[selected_label] = cur_direction
    app.logger.info(direction_per_cluster)
    locations = TSNE(n_components=2,init='random').fit_transform(direction_per_cluster)
    np.save('locations.npy',np.array(locations,dtype=int))
    np.save('labels.npy', labels)
    resp = defaultdict(list)
    for idx, i in enumerate(range(6)):
        representative_img = np.where(labels==i)[0]
        list_of_destinations = []
        for img_number in representative_img:
            list_of_destinations.append('ganzilla_images/test/' + str(all_seeds[0]) + '_' + str(number_of_clicks-1) + '/img' + str(img_number) +'.png')
        resp[i] = [int(locations[idx][0]), int(locations[idx][1]), list_of_destinations]
        app.logger.info(list_of_destinations)        
    #locations = locations.tolist()
    resp = json.dumps(resp)
    resp = make_response(resp)
    return resp


@app.route("/save-as-binary/", methods=['POST'])
def binary_saver():
    with open('public/state.txt', 'r') as f:
        data = json.load(f)
    cur_x = 0
    while os.path.isfile('public/ganzilla_images/initial_images/img' + str(cur_x) + '_' + str(data['y']) + '.png'):
        cur_x += 1


    filename = 'public/ganzilla_images/initial_images/img' + str(cur_x) + '_' + str(data['y']) + '.png'
    filepath = os.path.join('', filename)
    request.files['image'].save(filepath)
    app.logger.info(filepath)

    with open('public/state.txt', 'r') as f:
        json_data = json.load(f)
        json_data['highlighted'] = cur_x+1

    with open('public/state.txt', 'w') as f:
        f.write(json.dumps(json_data)) 
    return jsonify({})

@app.route('/update_x', methods=['GET', 'POST'])
def update_x():
    #This is the first scatter, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':
        datafromjs = request.form['mydata']
        updated_x = yaml.load(datafromjs)
        app.logger.info(updated_x)

        with open('public/actions.txt', 'a') as file:
            file.write(str(time.time()) + ',' + str('update_x_select_image') + ',' + str(updated_x) + '\n')
        

        with open('public/state.txt', 'r') as f:
            json_data = json.load(f)
            json_data['x'] = updated_x

        with open('public/state.txt', 'w') as f:
            f.write(json.dumps(json_data))


            
    return jsonify(True)

@app.route('/number_cluster', methods=['GET', 'POST'])
def number_cluster():
    #This is the first scatter, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':
        datafromjs = request.form['mydata']
        number_cluster = yaml.load(datafromjs)

        with open('public/state.txt', 'r') as f:
            json_data = json.load(f)
            json_data['cluster'] = number_cluster
        
        with open('public/actions.txt', 'a') as file:
            file.write(str(time.time()) + ',' + str('number_cluster') + ',' + str(number_cluster) + '\n')


        path_to_old = 'public/ganzilla_images/raw_images/' + str(json_data['x']) + '_' + str(json_data['y']) + '/' + str(json_data['depth']) + '/' + str(json_data['breadth']) + '/'
        #correct from here
        #Cluster these images
        img_names = []
        img_featues = []
        max_img_id = 0
        path_to_old_cp = 'public/ganzilla_images/raw_images/' + str(json_data['x']) + '_' + str(json_data['y']) + '/'
        with open(path_to_old_cp + "children.pkl", "rb") as pkl_handle:
	        children = pickle.load(pkl_handle)
        with open(path_to_old_cp + "parent.pkl", "rb") as pkl_handle:
	        parent = pickle.load(pkl_handle)
        for element in os.listdir(path_to_old):
            if '.png' in element and 'test' not in element:
                image_id = int(element[:-4])
                if image_id > max_img_id:
                    max_img_id = image_id

        for idx, image in enumerate(range(max_img_id+1)): #fix this, so it works for everything and create a function
            img_p = path_to_old + str(image)+'.png'
            app.logger.info(img_p)
            with Image.open(img_p) as im:
                image_input = preprocess(im).unsqueeze(0).to(device)
                text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
        

            # Calculate features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                #text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_featues.append(image_features)
            img_names.append(img_p)        


        #clustering
        img_features = torch.cat(img_featues)
        img_tmp = np.array(img_names)
        n_clusters = json_data['cluster']
        kmeans = KMeans(n_clusters=n_clusters).fit(img_features.cpu())
        labels = kmeans.labels_
        with open(path_to_old + 'directions_list.npy', 'rb') as f:
            directions_list = np.load(f)

        average_embeddings = np.zeros((n_clusters,directions_list.shape[1]))
        direction_per_cluster = np.zeros((n_clusters,directions_list.shape[1]))
        for label in range(n_clusters):
            chosen_direction_per_cluster = directions_list[np.where(labels==label)]
            average_embeddings[label] = np.mean(chosen_direction_per_cluster,0) #average embedding for a cluster
        for selected_label in range(n_clusters):
            chosen_cluster_average_direction = average_embeddings[selected_label]
            cur_direction = 0
            for label in range(n_clusters):
                if label != selected_label:
                    cur_cluster_average_direction = average_embeddings[label]
                    tmp_direction_current = chosen_cluster_average_direction - cur_cluster_average_direction
                    tmp_direction_current = tmp_direction_current/np.linalg.norm(tmp_direction_current)
                    cur_direction += tmp_direction_current
            cur_direction = cur_direction/np.linalg.norm(cur_direction)
            direction_per_cluster[selected_label] = cur_direction

        locations = TSNE(n_components=2,init='random').fit_transform(direction_per_cluster)

        center_x = json_data['center_x']
        center_y = json_data['center_y']
        avg_distance = json_data['avg_dist']
        #mean location
        mean_loc = np.mean(locations,0)
        distance_to_center = 0
        min_distance_to_center = 9999
        max_distance_to_center = 0
        for i in range(n_clusters):
            #move to center
            locations[i][0] = center_x - mean_loc[0] + locations[i][0]
            locations[i][1] = center_y - mean_loc[1] + locations[i][1]

            cur_dist = np.sqrt((locations[i][0] - center_x)**2 + (locations[i][1] - center_y)**2)
            distance_to_center += cur_dist
            if min_distance_to_center > cur_dist:
                min_distance_to_center = cur_dist
            if max_distance_to_center < cur_dist:
                max_distance_to_center = cur_dist


        app.logger.info(max_distance_to_center)
        distance_to_center /= n_clusters*(n_clusters-1) #average_distance to center
        ratio_of_dist = avg_distance/max_distance_to_center #ratio to multiply with 
        for i in range(n_clusters):
            #vector from center location to all the clusters
            locations[i][0] = (ratio_of_dist+1)*center_x - ratio_of_dist*locations[i][0]
            locations[i][1] = (ratio_of_dist+1)*center_y - ratio_of_dist*locations[i][1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

        np.save('locations.npy',np.array(locations,dtype=int))
        np.save('labels.npy', labels)

        np.save(path_to_old+'locations.npy',np.array(locations,dtype=int))
        np.save(path_to_old+'labels.npy', labels)
        resp = defaultdict(list)
        cur_children = children[str(json_data['depth']) + '/' + str(json_data['breadth'])]
        cur_parent = parent[str(json_data['depth']) + '/' + str(json_data['breadth'])]
        for idx, i in enumerate(range(n_clusters)):
            representative_img = np.where(labels==i)[0]
            list_of_destinations = []
            for img_number in representative_img:
                list_of_destinations.append(path_to_old[7:] + str(img_number) +'.png')
            resp[i] = [int(locations[idx][0]), int(locations[idx][1]), list_of_destinations]        
        #locations = locations.tolist()
        with open('public/state.txt', 'w') as f:
            f.write(json.dumps(json_data))
        resp[n_clusters] = ([len(cur_parent),len(cur_children)])
        resp = json.dumps(resp)
        with open(path_to_old + 'render.txt', 'w') as f:
            f.write(resp)
        resp = make_response(resp)
    return resp

@app.route('/get_image', methods=['GET', 'POST'])
def get_image():
    #This is the first scatter, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':
        datafromjs = request.form['mydata']
        image_seed = yaml.load(datafromjs)
        app.logger.info('pogchamp')
        with open('public/actions.txt', 'a') as file:
            file.write(str(time.time()) + ',' + 'get_image' + ',' + str(image_seed) +'\n')

    return jsonify('ganzilla_images/initial_images/generated' + str(image_seed) + '.png')

@app.route('/highlighter_opens', methods=['GET', 'POST'])
def highlighter_opens():
    with open('public/state.txt', 'r') as f:
        json_data = json.load(f)
        json_data['highlighted'] = 0
        all_seeds = json_data['seed']

    with open('public/state.txt', 'w') as f:
        f.write(json.dumps(json_data)) 

    with open('public/actions.txt', 'a') as file:
        file.write(str(time.time()) + ',' + 'highlighter_open' + '\n')
    OG = sty.Generator(G.z_dim,G .c_dim,G.w_dim,G.img_resolution,G.img_channels, synthesis_kwargs=args.G_kwargs.synthesis_kwargs).cuda()
    OG.load_state_dict(G.state_dict(), strict=False)
    OG.eval()
    for layer in json_data['layers']:
        if layer != 4:
            cur_block = getattr(OG.styling, 'b' + str(layer))
            cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
            for idx, affine in enumerate(affines):
                cur_aff = getattr(cur_block, affine)
                cur_aff_target = getattr(cur_block_target, 'conv' + str(idx))
                cur_aff_target = getattr(cur_aff_target,'affine')
                for weight in weights: # G.synthesis.b128.conv1.affine.weight
                    setattr(cur_aff, weight, getattr(cur_aff_target,weight))
        else:
            cur_block = getattr(OG.styling, 'b' + str(layer))
            cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
            cur_aff = getattr(cur_block, 'affine1')
            cur_aff_target = getattr(cur_block_target, 'conv1')
            cur_aff_target = getattr(cur_aff_target,'affine')
            for weight in weights: # G.synthesis.b128.conv1.affine.weight
                setattr(cur_aff, weight, getattr(cur_aff_target,weight))

    pathh='public/ganzilla_images/'
    if not os.path.exists(pathh):
        os.mkdir(pathh)
    pathh='public/ganzilla_images/initial_images/'
    if not os.path.exists(pathh):
        os.mkdir(pathh)
    for image_idx, image_seed in enumerate(all_seeds):
        all_z = np.stack([np.random.RandomState(image_seed).randn(G.z_dim)])
        all_w = OG.mapping(torch.from_numpy(all_z).to(device), None)
        styles = OG.styling(all_w)
        all_images = OG.synthesis(all_w, styles, noise_mode='const') #found images and filters
        all_images = OG.synthesis(all_w, styles, noise_mode='const')
        all_images_tmp = (all_images[0].clamp(-1, 1) + 1) / 2
        save_image(all_images_tmp[0],'public/ganzilla_images/initial_images/generated' + str(image_idx) + '.png')
    return jsonify({})

@app.route('/selector_opens', methods=['GET', 'POST'])
def selector_opens():

    with open('public/actions.txt', 'a') as file:
        file.write(str(time.time()) + ',' + 'selector_open' + '\n')
    with open('public/state.txt', 'r') as f:
        json_data = json.load(f)

        if json_data['highlighted'] == 0:
            step = 0
            images = []
            while (step < len(json_data['seed'])):
                images.append('ganzilla_images/initial_images/generated' + str(step) + '.png')
                step += 1
            pass_page = images
        else:
            step = 0
            images = []
            while (step < json_data['highlighted']):
                images.append('ganzilla_images/initial_images/img' + str(step) + '_' + str(json_data['y']) + '.png')
                step += 1
            while (step <  len(json_data['seed'])):
                images.append('ganzilla_images/initial_images/generated' + str(step) + '.png')
                step += 1
            pass_page = images
        resp = json.dumps(pass_page)
        resp = make_response(resp)
    return resp

@app.route('/ganzilla_opens', methods=['GET', 'POST'])
def ganzilla_opens():
    #This is the first scatter, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':
        with open('public/state.txt', 'r') as f:
            json_data = json.load(f)
            json_data['cluster'] = 6
            json_data['depth'] = 0
            json_data['breadth'] = 0
        with open('public/state.txt', 'w') as f:
            f.write(json.dumps(json_data))
        with open('public/actions.txt', 'a') as file:
            file.write(str(time.time()) + ',' + 'ganzilla_open' + '\n')
        #make dir for the public/ganzilla_images/raw_images/x_y/depth/breadth/
        path = 'public/ganzilla_images/raw_images/' + str(json_data['x']) + '_' + str(json_data['y']) + '/' + str(json_data['depth']) + '/' + str(json_data['breadth']) + '/'

        if not os.path.exists(path): #created for selected x and y, 0 depth 0 breadth
            os.makedirs(path)

        else:
            files = glob.glob(path+'*')
            for f in files:
                os.remove(f)
        children = defaultdict(list)
        parent = defaultdict(list)
        path_to_xy = 'public/ganzilla_images/raw_images/' + str(json_data['x']) + '_' + str(json_data['y']) + '/'
        with open(path_to_xy + "children.pkl", "wb") as pkl_handle:
	        pickle.dump(children, pkl_handle)
        with open(path_to_xy + "parent.pkl", "wb") as pkl_handle:
	        pickle.dump(parent, pkl_handle)
        #now select filters
        number_of_examples = json_data['examples']
        OG = sty.Generator(G.z_dim,G .c_dim,G.w_dim,G.img_resolution,G.img_channels, synthesis_kwargs=args.G_kwargs.synthesis_kwargs).cuda()
        OG.load_state_dict(G.state_dict(), strict=False)
        OG.eval()
        for layer in json_data['layers']:
            if layer != 4:
                cur_block = getattr(OG.styling, 'b' + str(layer))
                cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
                for idx, affine in enumerate(affines):
                    cur_aff = getattr(cur_block, affine)
                    cur_aff_target = getattr(cur_block_target, 'conv' + str(idx))
                    cur_aff_target = getattr(cur_aff_target,'affine')
                    for weight in weights: # G.synthesis.b128.conv1.affine.weight
                        setattr(cur_aff, weight, getattr(cur_aff_target,weight))
            else:
                cur_block = getattr(OG.styling, 'b' + str(layer))
                cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
                cur_aff = getattr(cur_block, 'affine1')
                cur_aff_target = getattr(cur_block_target, 'conv1')
                cur_aff_target = getattr(cur_aff_target,'affine')
                for weight in weights: # G.synthesis.b128.conv1.affine.weight
                    setattr(cur_aff, weight, getattr(cur_aff_target,weight))


        if json_data['highlighted'] > 0:
            real_selected_filters = []
            #for image_idx, seed in enumerate(json_data['seed']):
            for image_idx in range(json_data['highlighted']):
                seed = json_data['seed'][image_idx]
                all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim)])
                all_w = OG.mapping(torch.from_numpy(all_z).to(device), None)
                styles = OG.styling(all_w)
                all_images = OG.synthesis(all_w, styles, noise_mode='const') #found images and filters
                I = np.asarray(Image.open('public/ganzilla_images/initial_images/img' + str(image_idx) + '_' + str(json_data['y']) +  '.png'))
                mask = np.mean(I[:,:,:3],2) == 0
                mask = torch.from_numpy(mask).to(dtype=torch.float32).cuda()
                selected_filters = []
                threshold = json_data['threshold'][image_idx] #get the threshold from json
                
                for i in range(len(json_data['layers'])):
                    if i == 0:
                        conv1 = all_images[1][i][1][0]
                        conv1 = F.interpolate(conv1[:,None,:,:], size=(image_size,image_size))
                        conv1 = torch.squeeze(conv1)
                        overlap1 = torch.mean(mask * torch.abs(conv1), axis=(1,2))
                        overlap1 = overlap1 / torch.mean(mask)
                        selected_filters.append((None,~(overlap1 > threshold)))
                    else:
                        conv0 = all_images[1][i][0][0]
                        conv0 = F.interpolate(conv0[:,None,:,:], size=(image_size,image_size))
                        conv0 = torch.squeeze(conv0)
                        overlap0 = torch.mean(mask * torch.abs(conv0), axis=(1,2))
                        overlap0 = overlap0 / torch.mean(mask)
                        conv1 = (all_images[1][i][1][0])
                        conv1 = F.interpolate(conv1[:,None,:,:], size=(image_size,image_size))
                        conv1 = torch.squeeze(conv1)
                        overlap1 = torch.mean(mask * torch.abs(conv1), axis=(1,2))
                        overlap1 = overlap1 / torch.mean(mask)
                        selected_filters.append((~(overlap0 > threshold), ~(overlap1 > threshold)))
                real_selected_filters.append(selected_filters)
        
        else:
            real_selected_filters = []
            seed = json_data['seed'][0]
            all_z = np.stack([np.random.RandomState(seed).randn(G.z_dim)])
            all_w = OG.mapping(torch.from_numpy(all_z).to(device), None)
            styles = OG.styling(all_w)
            all_images = OG.synthesis(all_w, styles, noise_mode='const') #found images and filters
            I = np.asarray(Image.open('public/ganzilla_images/initial_images/generated0.png'))
            mask = np.mean(I[:,:,:3],2) > 0 #to avoid 0 division
            mask = torch.from_numpy(mask).to(dtype=torch.float32).cuda()
            selected_filters = []
            threshold = float('-inf') #select all filters
            for i in range(len(json_data['layers'])):
                if i == 0:
                    conv1 = all_images[1][i][1][0]
                    conv1 = F.interpolate(conv1[:,None,:,:], size=(image_size,image_size))
                    conv1 = torch.squeeze(conv1)
                    overlap1 = torch.mean(mask * torch.abs(conv1), axis=(1,2))
                    overlap1 = overlap1 / torch.mean(mask)
                    selected_filters.append((None,~(overlap1 > threshold)))
                else:
                    conv0 = all_images[1][i][0][0]
                    conv0 = F.interpolate(conv0[:,None,:,:], size=(image_size,image_size))
                    conv0 = torch.squeeze(conv0)
                    overlap0 = torch.mean(mask * torch.abs(conv0), axis=(1,2))
                    overlap0 = overlap0 / torch.mean(mask)
                    conv1 = (all_images[1][i][1][0])
                    conv1 = F.interpolate(conv1[:,None,:,:], size=(image_size,image_size))
                    conv1 = torch.squeeze(conv1)
                    overlap1 = torch.mean(mask * torch.abs(conv1), axis=(1,2))
                    overlap1 = overlap1 / torch.mean(mask)
                    selected_filters.append((~(overlap0 > threshold), ~(overlap1 > threshold)))
            real_selected_filters.append(selected_filters)
            
        selected_filters = real_selected_filters[0]
        
        for s_filter in real_selected_filters:
            for layer_idx, layer in enumerate(s_filter):
                if layer_idx == 0:
                    selected_filters[layer_idx] = (None,selected_filters[layer_idx][1] * layer[1])
                else:
                    selected_filters[layer_idx] = ((selected_filters[layer_idx][0] * layer[0], selected_filters[layer_idx][1] * layer[1]))
                    
        for i in range(len(selected_filters)):
            if i == 0:
                selected_filters[i] = (None, ~(selected_filters[i][1] > 0)) 
            else:
                selected_filters[i] = (~(selected_filters[i][0]>0), ~(selected_filters[i][1]>0))         
        for idx, i in enumerate(selected_filters):
            if idx == 0:
                app.logger.info(i[1].sum())
            else:
                app.logger.info(i[0].sum())
                app.logger.info(i[1].sum())

        #create initial images
        directions = []
        all_seeds = json_data['seed']
        all_z = np.stack([np.random.RandomState(all_seeds[json_data['x']]).randn(G.z_dim)]) #selected image z
        all_w = OG.mapping(torch.from_numpy(all_z).to(device), None)
        subset_exp = json_data['subset_exp']
        increase = json_data['increase'][json_data['x']] #get the increase from json
        for i in range(number_of_examples):
            direction_of_image = []
            new_styles = OG.styling(all_w)
            styles = OG.styling(all_w)
            new_style = []
            for idx, selects in enumerate(selected_filters):
                if idx == 0:
                    sel_styles = styles[idx+1][0][:,selects[1]] #select 0,1 -> style 1,0
                    random_tmp = torch.randint(2, size = sel_styles.shape)
                    app.logger.info(torch.sum(random_tmp))
                    for subset_exp_val in range(subset_exp-1):
                        random_tmp = random_tmp * torch.randint(2, size = sel_styles.shape)
                    #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                    dir1 = sel_styles + (random_tmp*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                    new_styles[idx+1][0][:,selects[1]] = dir1
                    direction_of_image.append((None, dir1))
                elif idx <= len(json_data['layers'])-2:
                    sel_styles = styles[idx][1][:,selects[0]] # select 1,0 style 1,1
                    #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                    random_tmp2 = torch.randint(2, size = sel_styles.shape)
                    for subset_exp_val in range(subset_exp-1):
                        random_tmp2 = random_tmp2 * torch.randint(2, size = sel_styles.shape)
                    if idx == 2 or idx == 3:
                        dir0 = sel_styles + (random_tmp2*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*30*increase
                    else:
                        dir0 = sel_styles + (random_tmp2*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                    new_styles[idx][1][:,selects[0]] = dir0
                    sel_styles = styles[idx+1][0][:,selects[1]] #select 1,1 style 2,0
                    #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                    random_tmp3 = torch.randint(2, size = sel_styles.shape)
                    for subset_exp_val in range(subset_exp-1):
                        random_tmp3 = random_tmp3 * torch.randint(2, size = sel_styles.shape)
                    if idx == 2 or idx == 3:
                        dir1 = sel_styles + (random_tmp3*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*30*increase
                    else:
                        dir1 = sel_styles + (random_tmp3*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                    new_styles[idx+1][0][:,selects[1]] = dir1
                    direction_of_image.append((dir0, dir1))
                else:
                    sel_styles = styles[idx][1][:,selects[0]] # select 8,0 style 8,1
                    #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                    random_tmp4 = torch.randint(2, size = sel_styles.shape)
                    for subset_exp_val in range(subset_exp-1):
                        random_tmp4 = random_tmp4 * torch.randint(2, size = sel_styles.shape)
                    dir0 = sel_styles + (random_tmp4*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                    new_styles[idx][1][:,selects[0]] = dir0
                    direction_of_image.append((dir0, None))
            directions.append(direction_of_image)
            new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
            new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
            save_image(new_images_tmp[0], path +  str(i) +'.png')
        
        #Cluster these images
        img_names = []
        img_featues = []
        for idx, image in enumerate(range(number_of_examples)): #fix this, so it works for everything and create a function
            img_p = path + str(image)+'.png'
            app.logger.info(img_p)
            with Image.open(img_p) as im:
                image_input = preprocess(im).unsqueeze(0).to(device)
                text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
        

            # Calculate features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                #text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_featues.append(image_features)
            img_names.append(img_p)
        
        #Should be a function
        directions_list_tmp = []
        for sample_id,direction in enumerate(directions):
            dir_tmp = direction[0][1][0]
            for i in range(1,len(json_data['layers'])-1):
                dir_tmp = torch.cat((dir_tmp,direction[i][0][0],direction[i][1][0]))
            dir_tmp = torch.cat((dir_tmp,direction[len(json_data['layers'])-1][0][0]))
            directions_list_tmp.append(dir_tmp.unsqueeze(0))
        directions_list = torch.cat(directions_list_tmp)
        directions_list = directions_list.cpu().detach().numpy()
        
        np.save('directions_list.npy',np.array(directions_list))
        pickle.dump(directions, open( "directions.pkl", "wb" ) )
        pickle.dump(selected_filters, open ('selected_filters.pkl', 'wb'))

        np.save(path + 'directions_list.npy',np.array(directions_list))
        pickle.dump(directions, open("directions.pkl", "wb" ) )
    
        #clustering
        img_features = torch.cat(img_featues)
        img_tmp = np.array(img_names)
        n_clusters = json_data['cluster']
        kmeans = KMeans(n_clusters=n_clusters).fit(img_features.cpu())
        labels = kmeans.labels_

        average_embeddings = np.zeros((n_clusters,directions_list.shape[1]))
        direction_per_cluster = np.zeros((n_clusters,directions_list.shape[1]))
        for label in range(n_clusters):
            chosen_direction_per_cluster = directions_list[np.where(labels==label)]
            average_embeddings[label] = np.mean(chosen_direction_per_cluster,0) #average embedding for a cluster
        for selected_label in range(n_clusters):
            chosen_cluster_average_direction = average_embeddings[selected_label]
            cur_direction = 0
            for label in range(n_clusters):
                if label != selected_label:
                    cur_cluster_average_direction = average_embeddings[label]
                    tmp_direction_current = chosen_cluster_average_direction - cur_cluster_average_direction
                    tmp_direction_current = tmp_direction_current/np.linalg.norm(tmp_direction_current)
                    cur_direction += tmp_direction_current
            cur_direction = cur_direction/np.linalg.norm(cur_direction)
            direction_per_cluster[selected_label] = cur_direction

        locations = TSNE(n_components=2,init='random').fit_transform(direction_per_cluster)

        center_x = json_data['center_x']
        center_y = json_data['center_y']
        avg_distance = json_data['avg_dist']
        #mean location
        mean_loc = np.mean(locations,0)
        distance_to_center = 0
        min_distance_to_center = 9999
        max_distance_to_center = 0
        for i in range(n_clusters):
            #move to center
            locations[i][0] = center_x - mean_loc[0] + locations[i][0]
            locations[i][1] = center_y - mean_loc[1] + locations[i][1]

            cur_dist = np.sqrt((locations[i][0] - center_x)**2 + (locations[i][1] - center_y)**2)
            distance_to_center += cur_dist
            if min_distance_to_center > cur_dist:
                min_distance_to_center = cur_dist
            if max_distance_to_center < cur_dist:
                max_distance_to_center = cur_dist


        app.logger.info(max_distance_to_center)
        distance_to_center /= n_clusters*(n_clusters-1) #average_distance to center
        ratio_of_dist = avg_distance/max_distance_to_center #ratio to multiply with 
        for i in range(n_clusters):
            #vector from center location to all the clusters
            locations[i][0] = (ratio_of_dist+1)*center_x - ratio_of_dist*locations[i][0]
            locations[i][1] = (ratio_of_dist+1)*center_y - ratio_of_dist*locations[i][1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

        np.save('locations.npy',np.array(locations,dtype=int))
        np.save('labels.npy', labels)

        np.save(path+'locations.npy',np.array(locations,dtype=int))
        np.save(path+'labels.npy', labels)
        resp = defaultdict(list)
        cur_children = children[str(json_data['depth']) + '/' + str(json_data['breadth'])]
        cur_parent = parent[str(json_data['depth']) + '/' + str(json_data['breadth'])]
        for idx, i in enumerate(range(n_clusters)):
            representative_img = np.where(labels==i)[0]
            list_of_destinations = []
            for img_number in representative_img:
                list_of_destinations.append(path[7:] + str(img_number) +'.png')
            resp[i] = [int(locations[idx][0]), int(locations[idx][1]), list_of_destinations]        
        #locations = locations.tolist()
        resp[n_clusters] = ([len(cur_parent),len(cur_children)])
        resp = json.dumps(resp)
        with open(path + 'render.txt', 'w') as f:
            f.write(resp)
        resp = make_response(resp)
        return resp

@app.route('/testis', methods=['GET', 'POST'])
def scatter():
    #This is the other scatters, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':
        datafromjs = request.form['mydata']
        selected = yaml.load(datafromjs)
        app.logger.info(selected)

        with open('public/state.txt', 'r') as f:
            json_data = json.load(f)
            subset_exp = json_data['subset_exp']

        path_to_old = 'public/ganzilla_images/raw_images/' + str(json_data['x']) + '_' + str(json_data['y']) + '/' + str(json_data['depth']) + '/' + str(json_data['breadth']) + '/'
        number_of_examples = json_data['examples']
        all_seeds = json_data['seed']
        selected_image = json_data['x']
        current_depth = json_data['depth']
        current_breadth = json_data['breadth']
        #figure out current parents, children
        path_to_old_cp = 'public/ganzilla_images/raw_images/' + str(json_data['x']) + '_' + str(json_data['y']) + '/'
        # LOAD
        with open(path_to_old_cp + "children.pkl", "rb") as pkl_handle:
	        children = pickle.load(pkl_handle)
        with open(path_to_old_cp + "parent.pkl", "rb") as pkl_handle:
	        parent = pickle.load(pkl_handle)

        #latest directions
        with open(path_to_old + 'directions_list.npy', 'rb') as f:
            directions_list = np.load(f)
        with open(path_to_old + 'labels.npy', 'rb') as f:
            labels = np.load(f)
        
        selected_filters = pickle.load( open('selected_filters.pkl', "rb" ) )
        directions = pickle.load( open('directions.pkl', "rb" ) )
        OG = sty.Generator(G.z_dim,G .c_dim,G.w_dim,G.img_resolution,G.img_channels, synthesis_kwargs=args.G_kwargs.synthesis_kwargs).cuda()
        OG.load_state_dict(G.state_dict(), strict=False)
        OG.eval()
        for layer in json_data['layers']:
            if layer != 4:
                cur_block = getattr(OG.styling, 'b' + str(layer))
                cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
                for idx, affine in enumerate(affines):
                    cur_aff = getattr(cur_block, affine)
                    cur_aff_target = getattr(cur_block_target, 'conv' + str(idx))
                    cur_aff_target = getattr(cur_aff_target,'affine')
                    for weight in weights: # G.synthesis.b128.conv1.affine.weight
                        setattr(cur_aff, weight, getattr(cur_aff_target,weight))
            else:
                cur_block = getattr(OG.styling, 'b' + str(layer))
                cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
                cur_aff = getattr(cur_block, 'affine1')
                cur_aff_target = getattr(cur_block_target, 'conv1')
                cur_aff_target = getattr(cur_aff_target,'affine')
                for weight in weights: # G.synthesis.b128.conv1.affine.weight
                    setattr(cur_aff, weight, getattr(cur_aff_target,weight))

        all_z = np.stack([np.random.RandomState(all_seeds[json_data['x']]).randn(G.z_dim)]) #selected image z
        all_w = OG.mapping(torch.from_numpy(all_z).to(device), None)
        new_styles = OG.styling(all_w)
        increase = json_data['increase'][json_data['x']] #get the increase from json
        direction = directions[0]

        if np.array(selected).sum() == 0:
            with open('public/actions.txt', 'a') as file:
                file.write(str(time.time()) + ',' + 'more' + ',' + str(selected) + '\n')
            app.logger.info('nothing selected')
            new_path = path_to_old
            
            #find the number of images in path_to_old

            max_img_id = 0
            for element in os.listdir(path_to_old):
                if '.png' in element and 'test' not in element:
                    image_id = int(element[:-4])
                    if image_id > max_img_id:
                        max_img_id = image_id
            app.logger.info(max_img_id)
            #sample more images
            if current_depth == 0:
                for i in range(number_of_examples):
                    direction_of_image = []
                    new_styles = OG.styling(all_w)
                    styles = OG.styling(all_w)
                    new_style = []
                    for idx, selects in enumerate(selected_filters):
                        if idx == 0:
                            sel_styles = styles[idx+1][0][:,selects[1]] #select 0,1 -> style 1,0
                            random_tmp = torch.randint(2, size = sel_styles.shape)
                            app.logger.info(torch.sum(random_tmp))
                            for subset_exp_val in range(subset_exp-1):
                                random_tmp = random_tmp * torch.randint(2, size = sel_styles.shape)
                            #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                            dir1 = sel_styles + (random_tmp*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                            new_styles[idx+1][0][:,selects[1]] = dir1
                            direction_of_image.append((None, dir1))
                        elif idx <= len(json_data['layers'])-2:
                            sel_styles = styles[idx][1][:,selects[0]] # select 1,0 style 1,1
                            #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                            random_tmp2 = torch.randint(2, size = sel_styles.shape)
                            for subset_exp_val in range(subset_exp-1):
                                random_tmp2 = random_tmp2 * torch.randint(2, size = sel_styles.shape)
                            if idx == 2 or idx == 3:
                                dir0 = sel_styles + (random_tmp2*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*30*increase
                            else:
                                dir0 = sel_styles + (random_tmp2*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                            new_styles[idx][1][:,selects[0]] = dir0
                            sel_styles = styles[idx+1][0][:,selects[1]] #select 1,1 style 2,0
                            #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                            random_tmp3 = torch.randint(2, size = sel_styles.shape)
                            for subset_exp_val in range(subset_exp-1):
                                random_tmp3 = random_tmp3 * torch.randint(2, size = sel_styles.shape)
                            if idx == 2 or idx == 3:
                                dir1 = sel_styles + (random_tmp3*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*30*increase
                            else:
                                dir1 = sel_styles + (random_tmp3*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                            new_styles[idx+1][0][:,selects[1]] = dir1
                            direction_of_image.append((dir0, dir1))
                        else:
                            sel_styles = styles[idx][1][:,selects[0]] # select 8,0 style 8,1
                            #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                            random_tmp4 = torch.randint(2, size = sel_styles.shape)
                            for subset_exp_val in range(subset_exp-1):
                                random_tmp4 = random_tmp4 * torch.randint(2, size = sel_styles.shape)
                            dir0 = sel_styles + (random_tmp4*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                            new_styles[idx][1][:,selects[0]] = dir0
                            direction_of_image.append((dir0, None))
                    directions.append(direction_of_image)
                    new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
                    new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
                    save_image(new_images_tmp[0], path_to_old +  str(i+max_img_id+1) +'.png')
                    pickle.dump(directions, open("directions.pkl", "wb" ) )

                    directions_list_tmp = []
                    for sample_id,direction in enumerate(directions):
                        dir_tmp = direction[0][1][0]
                        for i in range(1,len(json_data['layers'])-1):
                            dir_tmp = torch.cat((dir_tmp,direction[i][0][0],direction[i][1][0]))
                        dir_tmp = torch.cat((dir_tmp,direction[len(json_data['layers'])-1][0][0]))
                        directions_list_tmp.append(dir_tmp.unsqueeze(0))
                    directions_list = torch.cat(directions_list_tmp)
                    directions_list = directions_list.cpu().detach().numpy()

                    
            else:
                with open(path_to_old + 'direction_subset.npy', 'rb') as f:
                        direction_subset = np.load(f)
                directions_list_new = np.zeros((number_of_examples,directions_list.shape[1]))
                increase = json_data['increase'][json_data['x']]
                for cur_example_id in range(number_of_examples):
                    dir_tmp = []
                    chsn = direction[0][1][0].shape[0]
                    pc_values = random.choice(direction_subset)
                    pc_values2 = random.choice(direction_subset)
                    pc_values = (pc_values+pc_values2)/2
                    app.logger.info('hereeeeeeeeeeeeeeeeeeee')
                    pc_values += (np.random.rand(pc_values.shape[0])-0.125/2)*10*increase*np.random.randint(2, size = pc_values.shape[0])*np.random.randint(2, size = pc_values.shape[0])*np.random.randint(2, size = pc_values.shape[0])*np.random.randint(2, size = pc_values.shape[0])
                    
                    directions_list_new[cur_example_id] = pc_values
                    dir_tmp.append((None,torch.tensor(pc_values[:chsn]).float()))
                    for i in range(1,len(json_data['layers'])-1):
                        chsn0 = direction[i][0][0].shape[0]
                        chsn1 = direction[i][1][0].shape[0]
                        dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn0]).float(),torch.tensor(pc_values[chsn+chsn0:chsn+chsn0+chsn1]).float()))
                        chsn += chsn0 + chsn1
                    chsn1 = direction[len(json_data['layers'])-1][0][0].shape[0]
                    dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn1]).float(),None))

                    new_styles = OG.styling(all_w)
                    for idx, selects in enumerate(selected_filters):
                        if idx == 0:
                            dir1 = dir_tmp[0][1]
                            new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda()) 

                        elif idx <= len(json_data['layers'])-2:
                            if idx == 2 or idx == 3:
                                dir0 = dir_tmp[idx][0]
                            else:
                                dir0 = dir_tmp[idx][0]
                            new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())

                            if idx == 2 or idx == 3:
                                dir1 = dir_tmp[idx][1]
                            else:
                                dir1 = dir_tmp[idx][1]
                            new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda())


                        else:
                            dir0 = dir_tmp[idx][0]
                            new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())   
                    new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
                    new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
                    save_image(new_images_tmp[0], path_to_old + str(cur_example_id+max_img_id+1) +'.png')
                
                directions_list = np.concatenate((directions_list,directions_list_new),0)


            


        else: #when something is selectedsaved_direction_finish y, depth breadth
            with open('public/actions.txt', 'a') as file:
                file.write(str(time.time()) + ',' + 'scatter' + ',' + str(selected) + '\n')
            new_path_tmp = 'public/ganzilla_images/raw_images/' + str(json_data['x']) + '_' + str(json_data['y']) + '/' + str(json_data['depth']+1) + '/'
            if not os.path.exists(new_path_tmp): #created for selected x and y, depth breadth
                os.makedirs(new_path_tmp)
            
            biggest_breadth = -1
            for breadths in os.listdir(new_path_tmp):
                if biggest_breadth < int(breadths):
                    biggest_breadth = int(breadths)
            new_path = new_path_tmp + str(biggest_breadth+1) + '/'
            if not os.path.exists(new_path): #created for selected x and y, depth breadth
                os.makedirs(new_path)
            old_breadth = json_data['breadth']
            old_depth = json_data['depth']
            json_data['breadth'] = biggest_breadth+1
            json_data['depth'] += 1

            children[str(old_depth) + '/' + str(old_breadth)].append(str(json_data['depth']) + '/' + str(json_data['breadth']))
            parent[str(json_data['depth']) + '/' + str(json_data['breadth'])].append(str(old_depth) + '/' + str(old_breadth))
            with open(path_to_old_cp + "children.pkl", "wb") as pkl_handle:
                pickle.dump(children, pkl_handle)
            with open(path_to_old_cp + "parent.pkl", "wb") as pkl_handle:
                pickle.dump(parent, pkl_handle)


            with open(path_to_old + 'render.txt', 'r') as f:
                render_data = json.load(f)
            
            render_button_idx = list(render_data.keys())[-1]
            num_next = len(children[str(old_depth) + '/' + str(old_breadth)])
            num_prev = len(parent[str(old_depth) + '/' + str(old_breadth)])
            render_data[render_button_idx] = [num_prev, num_next]

            with open(path_to_old + 'render.txt', 'w') as f:
                f.write(json.dumps(render_data))
            app.logger.info(np.array(selected).sum())
            direction_subset = []
            for select_id,selects in enumerate(np.array(selected)):
                if selects:
                    app.logger.info(select_id)
                    direction_indices = np.where(labels == select_id)
                    for cur in directions_list[direction_indices]:
                        direction_subset.append(cur)
            direction_subset = np.array(direction_subset) 
            

            np.save(new_path + 'direction_subset.npy',np.array(direction_subset))        

            
            
            increase = json_data['increase'][json_data['x']]
            directions_list_new = np.zeros((number_of_examples,directions_list.shape[1]))
            for cur_example_id in range(number_of_examples):
                dir_tmp = []
                chsn = direction[0][1][0].shape[0]
                pc_values = random.choice(direction_subset)
                pc_values2 = random.choice(direction_subset)
                pc_values = (pc_values+pc_values2)/2
                app.logger.info('hereeeeeeeeeeeeeeeeeeee')
                pc_values += (np.random.rand(pc_values.shape[0])-0.125/2)*10*increase*np.random.randint(2, size = pc_values.shape[0])*np.random.randint(2, size = pc_values.shape[0])*np.random.randint(2, size = pc_values.shape[0])*np.random.randint(2, size = pc_values.shape[0])
                
                directions_list_new[cur_example_id] = pc_values
                dir_tmp.append((None,torch.tensor(pc_values[:chsn]).float()))
                for i in range(1,len(json_data['layers'])-1):
                    chsn0 = direction[i][0][0].shape[0]
                    chsn1 = direction[i][1][0].shape[0]
                    dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn0]).float(),torch.tensor(pc_values[chsn+chsn0:chsn+chsn0+chsn1]).float()))
                    chsn += chsn0 + chsn1
                chsn1 = direction[len(json_data['layers'])-1][0][0].shape[0]
                dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn1]).float(),None))

                new_styles = OG.styling(all_w)
                for idx, selects in enumerate(selected_filters):
                    if idx == 0:
                        dir1 = dir_tmp[0][1]
                        new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda()) 

                    elif idx <= len(json_data['layers'])-2:
                        if idx == 2 or idx == 3:
                            dir0 = dir_tmp[idx][0]
                        else:
                            dir0 = dir_tmp[idx][0]
                        new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())

                        if idx == 2 or idx == 3:
                            dir1 = dir_tmp[idx][1]
                        else:
                            dir1 = dir_tmp[idx][1]
                        new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda())


                    else:
                        dir0 = dir_tmp[idx][0]
                        new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())   
                new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
                new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
                save_image(new_images_tmp[0], new_path + str(cur_example_id) +'.png')
            #Should be a function
            directions_list = directions_list_new
            

        #correct from here
        #Cluster these images
        img_names = []
        img_featues = []
        max_img_id = 0
        for element in os.listdir(new_path):
            if '.png' in element and 'test' not in element:
                image_id = int(element[:-4])
                if image_id > max_img_id:
                    max_img_id = image_id

        for idx, image in enumerate(range(max_img_id+1)): #fix this, so it works for everything and create a function
            img_p = new_path + str(image)+'.png'
            app.logger.info(img_p)
            with Image.open(img_p) as im:
                image_input = preprocess(im).unsqueeze(0).to(device)
                text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
        

            # Calculate features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                #text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_featues.append(image_features)
            img_names.append(img_p)
        

        
        np.save('directions_list.npy',np.array(directions_list))
        pickle.dump(selected_filters, open ('selected_filters.pkl', 'wb'))

        np.save(new_path + 'directions_list.npy',np.array(directions_list))
            
        #clustering
        img_features = torch.cat(img_featues)
        img_tmp = np.array(img_names)
        n_clusters = json_data['cluster']
        kmeans = KMeans(n_clusters=n_clusters).fit(img_features.cpu())
        labels = kmeans.labels_

        average_embeddings = np.zeros((n_clusters,directions_list.shape[1]))
        direction_per_cluster = np.zeros((n_clusters,directions_list.shape[1]))
        for label in range(n_clusters):
            chosen_direction_per_cluster = directions_list[np.where(labels==label)]
            average_embeddings[label] = np.mean(chosen_direction_per_cluster,0) #average embedding for a cluster
        for selected_label in range(n_clusters):
            chosen_cluster_average_direction = average_embeddings[selected_label]
            cur_direction = 0
            for label in range(n_clusters):
                if label != selected_label:
                    cur_cluster_average_direction = average_embeddings[label]
                    tmp_direction_current = chosen_cluster_average_direction - cur_cluster_average_direction
                    tmp_direction_current = tmp_direction_current/np.linalg.norm(tmp_direction_current)
                    cur_direction += tmp_direction_current
            cur_direction = cur_direction/np.linalg.norm(cur_direction)
            direction_per_cluster[selected_label] = cur_direction

        locations = TSNE(n_components=2,init='random').fit_transform(direction_per_cluster)

        center_x = json_data['center_x']
        center_y = json_data['center_y']
        avg_distance = json_data['avg_dist']
        #mean location
        mean_loc = np.mean(locations,0)
        distance_to_center = 0
        min_distance_to_center = 9999
        max_distance_to_center = 0
        for i in range(n_clusters):
            #move to center
            locations[i][0] = center_x - mean_loc[0] + locations[i][0]
            locations[i][1] = center_y - mean_loc[1] + locations[i][1]

            cur_dist = np.sqrt((locations[i][0] - center_x)**2 + (locations[i][1] - center_y)**2)
            distance_to_center += cur_dist
            if min_distance_to_center > cur_dist:
                min_distance_to_center = cur_dist
            if max_distance_to_center < cur_dist:
                max_distance_to_center = cur_dist


        app.logger.info(max_distance_to_center)
        distance_to_center /= n_clusters*(n_clusters-1) #average_distance to center
        ratio_of_dist = avg_distance/max_distance_to_center #ratio to multiply with 
        for i in range(n_clusters):
            #vector from center location to all the clusters
            locations[i][0] = (ratio_of_dist+1)*center_x - ratio_of_dist*locations[i][0]
            locations[i][1] = (ratio_of_dist+1)*center_y - ratio_of_dist*locations[i][1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

        np.save('locations.npy',np.array(locations,dtype=int))
        np.save('labels.npy', labels)

        np.save(new_path+'locations.npy',np.array(locations,dtype=int))
        np.save(new_path+'labels.npy', labels)
        resp = defaultdict(list)
        cur_children = children[str(json_data['depth']) + '/' + str(json_data['breadth'])]
        cur_parent = parent[str(json_data['depth']) + '/' + str(json_data['breadth'])]
        for idx, i in enumerate(range(n_clusters)):
            representative_img = np.where(labels==i)[0]
            list_of_destinations = []
            for img_number in representative_img:
                list_of_destinations.append(new_path[7:] + str(img_number) +'.png')
            resp[i] = [int(locations[idx][0]), int(locations[idx][1]), list_of_destinations]        
        #locations = locations.tolist()
        with open('public/state.txt', 'w') as f:
            f.write(json.dumps(json_data))
        resp[n_clusters] = ([len(cur_parent),len(cur_children)])
        resp = json.dumps(resp)
        with open(new_path + 'render.txt', 'w') as f:
            f.write(resp)
        resp = make_response(resp)
        return resp

    return jsonify({})


@app.route('/get_more_images', methods=['GET', 'POST'])
def get_more_images():
    #This is the other scatters, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':
        with open('public/state.txt', 'r') as f:
            json_data = json.load(f)
            subset_exp = json_data['subset_exp']

        path_to_old = 'public/ganzilla_images/raw_images/' + str(json_data['x']) + '_' + str(json_data['y']) + '/' + str(json_data['depth']) + '/' + str(json_data['breadth']) + '/'
        number_of_examples = json_data['examples']
        all_seeds = json_data['seed']
        selected_image = json_data['x']
        current_depth = json_data['depth']
        current_breadth = json_data['breadth']
        #figure out current parents, children
        path_to_old_cp = 'public/ganzilla_images/raw_images/' + str(json_data['x']) + '_' + str(json_data['y']) + '/'
        # LOAD
        with open(path_to_old_cp + "children.pkl", "rb") as pkl_handle:
	        children = pickle.load(pkl_handle)
        with open(path_to_old_cp + "parent.pkl", "rb") as pkl_handle:
	        parent = pickle.load(pkl_handle)

        #latest directions
        with open(path_to_old + 'directions_list.npy', 'rb') as f:
            directions_list = np.load(f)
        with open(path_to_old + 'labels.npy', 'rb') as f:
            labels = np.load(f)
        
        selected_filters = pickle.load( open('selected_filters.pkl', "rb" ) )
        directions = pickle.load( open('directions.pkl', "rb" ) )
        OG = sty.Generator(G.z_dim,G .c_dim,G.w_dim,G.img_resolution,G.img_channels, synthesis_kwargs=args.G_kwargs.synthesis_kwargs).cuda()
        OG.load_state_dict(G.state_dict(), strict=False)
        OG.eval()
        for layer in json_data['layers']:
            if layer != 4:
                cur_block = getattr(OG.styling, 'b' + str(layer))
                cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
                for idx, affine in enumerate(affines):
                    cur_aff = getattr(cur_block, affine)
                    cur_aff_target = getattr(cur_block_target, 'conv' + str(idx))
                    cur_aff_target = getattr(cur_aff_target,'affine')
                    for weight in weights: # G.synthesis.b128.conv1.affine.weight
                        setattr(cur_aff, weight, getattr(cur_aff_target,weight))
            else:
                cur_block = getattr(OG.styling, 'b' + str(layer))
                cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
                cur_aff = getattr(cur_block, 'affine1')
                cur_aff_target = getattr(cur_block_target, 'conv1')
                cur_aff_target = getattr(cur_aff_target,'affine')
                for weight in weights: # G.synthesis.b128.conv1.affine.weight
                    setattr(cur_aff, weight, getattr(cur_aff_target,weight))

        all_z = np.stack([np.random.RandomState(all_seeds[json_data['x']]).randn(G.z_dim)]) #selected image z
        all_w = OG.mapping(torch.from_numpy(all_z).to(device), None)
        new_styles = OG.styling(all_w)
        increase = json_data['increase'][json_data['x']] #get the increase from json
        direction = directions[0]

        
        with open('public/actions.txt', 'a') as file:
            file.write(str(time.time()) + ',' + 'more' + ',' + '\n')
        app.logger.info('nothing selected')
        new_path = path_to_old
        
        #find the number of images in path_to_old

        max_img_id = 0
        for element in os.listdir(path_to_old):
            if '.png' in element and 'test' not in element:
                image_id = int(element[:-4])
                if image_id > max_img_id:
                    max_img_id = image_id
        app.logger.info(max_img_id)
        #sample more images
        if current_depth == 0:
            for i in range(number_of_examples):
                direction_of_image = []
                new_styles = OG.styling(all_w)
                styles = OG.styling(all_w)
                new_style = []
                for idx, selects in enumerate(selected_filters):
                    if idx == 0:
                        sel_styles = styles[idx+1][0][:,selects[1]] #select 0,1 -> style 1,0
                        random_tmp = torch.randint(2, size = sel_styles.shape)
                        app.logger.info(torch.sum(random_tmp))
                        for subset_exp_val in range(subset_exp-1):
                            random_tmp = random_tmp * torch.randint(2, size = sel_styles.shape)
                        #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                        dir1 = sel_styles + (random_tmp*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                        new_styles[idx+1][0][:,selects[1]] = dir1
                        direction_of_image.append((None, dir1))
                    elif idx <= len(json_data['layers'])-2:
                        sel_styles = styles[idx][1][:,selects[0]] # select 1,0 style 1,1
                        #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                        random_tmp2 = torch.randint(2, size = sel_styles.shape)
                        for subset_exp_val in range(subset_exp-1):
                            random_tmp2 = random_tmp2 * torch.randint(2, size = sel_styles.shape)
                        if idx == 2 or idx == 3:
                            dir0 = sel_styles + (random_tmp2*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*30*increase
                        else:
                            dir0 = sel_styles + (random_tmp2*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                        new_styles[idx][1][:,selects[0]] = dir0
                        sel_styles = styles[idx+1][0][:,selects[1]] #select 1,1 style 2,0
                        #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                        random_tmp3 = torch.randint(2, size = sel_styles.shape)
                        for subset_exp_val in range(subset_exp-1):
                            random_tmp3 = random_tmp3 * torch.randint(2, size = sel_styles.shape)
                        if idx == 2 or idx == 3:
                            dir1 = sel_styles + (random_tmp3*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*30*increase
                        else:
                            dir1 = sel_styles + (random_tmp3*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                        new_styles[idx+1][0][:,selects[1]] = dir1
                        direction_of_image.append((dir0, dir1))
                    else:
                        sel_styles = styles[idx][1][:,selects[0]] # select 8,0 style 8,1
                        #sel_styles = sel_styles[torch.randperm(sel_styles.shape[0]),:]
                        random_tmp4 = torch.randint(2, size = sel_styles.shape)
                        for subset_exp_val in range(subset_exp-1):
                            random_tmp4 = random_tmp4 * torch.randint(2, size = sel_styles.shape)
                        dir0 = sel_styles + (random_tmp4*torch.rand(size = sel_styles.shape)-(2**(-1*subset_exp))).cuda()*10*increase
                        new_styles[idx][1][:,selects[0]] = dir0
                        direction_of_image.append((dir0, None))
                directions.append(direction_of_image)
                new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
                new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
                save_image(new_images_tmp[0], path_to_old +  str(i+max_img_id+1) +'.png')
                pickle.dump(directions, open("directions.pkl", "wb" ) )

                directions_list_tmp = []
                for sample_id,direction in enumerate(directions):
                    dir_tmp = direction[0][1][0]
                    for i in range(1,len(json_data['layers'])-1):
                        dir_tmp = torch.cat((dir_tmp,direction[i][0][0],direction[i][1][0]))
                    dir_tmp = torch.cat((dir_tmp,direction[len(json_data['layers'])-1][0][0]))
                    directions_list_tmp.append(dir_tmp.unsqueeze(0))
                directions_list = torch.cat(directions_list_tmp)
                directions_list = directions_list.cpu().detach().numpy()

                
        else:
            with open(path_to_old + 'direction_subset.npy', 'rb') as f:
                    direction_subset = np.load(f)
            directions_list_new = np.zeros((number_of_examples,directions_list.shape[1]))
            increase = json_data['increase'][json_data['x']]
            for cur_example_id in range(number_of_examples):
                dir_tmp = []
                chsn = direction[0][1][0].shape[0]
                pc_values = random.choice(direction_subset)
                pc_values2 = random.choice(direction_subset)
                pc_values = (pc_values+pc_values2)/2
                app.logger.info('hereeeeeeeeeeeeeeeeeeee')
                pc_values += (np.random.rand(pc_values.shape[0])-0.125/2)*10*increase*np.random.randint(2, size = pc_values.shape[0])*np.random.randint(2, size = pc_values.shape[0])*np.random.randint(2, size = pc_values.shape[0])*np.random.randint(2, size = pc_values.shape[0])
                
                directions_list_new[cur_example_id] = pc_values
                dir_tmp.append((None,torch.tensor(pc_values[:chsn]).float()))
                for i in range(1,len(json_data['layers'])-1):
                    chsn0 = direction[i][0][0].shape[0]
                    chsn1 = direction[i][1][0].shape[0]
                    dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn0]).float(),torch.tensor(pc_values[chsn+chsn0:chsn+chsn0+chsn1]).float()))
                    chsn += chsn0 + chsn1
                chsn1 = direction[len(json_data['layers'])-1][0][0].shape[0]
                dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn1]).float(),None))

                new_styles = OG.styling(all_w)
                for idx, selects in enumerate(selected_filters):
                    if idx == 0:
                        dir1 = dir_tmp[0][1]
                        new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda()) 

                    elif idx <= len(json_data['layers'])-2:
                        if idx == 2 or idx == 3:
                            dir0 = dir_tmp[idx][0]
                        else:
                            dir0 = dir_tmp[idx][0]
                        new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())

                        if idx == 2 or idx == 3:
                            dir1 = dir_tmp[idx][1]
                        else:
                            dir1 = dir_tmp[idx][1]
                        new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda())


                    else:
                        dir0 = dir_tmp[idx][0]
                        new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())   
                new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
                new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
                save_image(new_images_tmp[0], path_to_old + str(cur_example_id+max_img_id+1) +'.png')
            
            directions_list = np.concatenate((directions_list,directions_list_new),0)

            

        #correct from here
        #Cluster these images
        img_names = []
        img_featues = []
        max_img_id = 0
        for element in os.listdir(new_path):
            if '.png' in element and 'test' not in element:
                image_id = int(element[:-4])
                if image_id > max_img_id:
                    max_img_id = image_id

        for idx, image in enumerate(range(max_img_id+1)): #fix this, so it works for everything and create a function
            img_p = new_path + str(image)+'.png'
            app.logger.info(img_p)
            with Image.open(img_p) as im:
                image_input = preprocess(im).unsqueeze(0).to(device)
                text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)
        

            # Calculate features
            with torch.no_grad():
                image_features = model.encode_image(image_input)
                #text_features = model.encode_text(text_inputs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            img_featues.append(image_features)
            img_names.append(img_p)
        

        
        np.save('directions_list.npy',np.array(directions_list))
        pickle.dump(selected_filters, open ('selected_filters.pkl', 'wb'))

        np.save(new_path + 'directions_list.npy',np.array(directions_list))
            
        #clustering
        img_features = torch.cat(img_featues)
        img_tmp = np.array(img_names)
        n_clusters = json_data['cluster']
        kmeans = KMeans(n_clusters=n_clusters).fit(img_features.cpu())
        labels = kmeans.labels_

        average_embeddings = np.zeros((n_clusters,directions_list.shape[1]))
        direction_per_cluster = np.zeros((n_clusters,directions_list.shape[1]))
        for label in range(n_clusters):
            chosen_direction_per_cluster = directions_list[np.where(labels==label)]
            average_embeddings[label] = np.mean(chosen_direction_per_cluster,0) #average embedding for a cluster
        for selected_label in range(n_clusters):
            chosen_cluster_average_direction = average_embeddings[selected_label]
            cur_direction = 0
            for label in range(n_clusters):
                if label != selected_label:
                    cur_cluster_average_direction = average_embeddings[label]
                    tmp_direction_current = chosen_cluster_average_direction - cur_cluster_average_direction
                    tmp_direction_current = tmp_direction_current/np.linalg.norm(tmp_direction_current)
                    cur_direction += tmp_direction_current
            cur_direction = cur_direction/np.linalg.norm(cur_direction)
            direction_per_cluster[selected_label] = cur_direction

        locations = TSNE(n_components=2,init='random').fit_transform(direction_per_cluster)

        center_x = json_data['center_x']
        center_y = json_data['center_y']
        avg_distance = json_data['avg_dist']
        #mean location
        mean_loc = np.mean(locations,0)
        distance_to_center = 0
        min_distance_to_center = 9999
        max_distance_to_center = 0
        for i in range(n_clusters):
            #move to center
            locations[i][0] = center_x - mean_loc[0] + locations[i][0]
            locations[i][1] = center_y - mean_loc[1] + locations[i][1]

            cur_dist = np.sqrt((locations[i][0] - center_x)**2 + (locations[i][1] - center_y)**2)
            distance_to_center += cur_dist
            if min_distance_to_center > cur_dist:
                min_distance_to_center = cur_dist
            if max_distance_to_center < cur_dist:
                max_distance_to_center = cur_dist


        app.logger.info(max_distance_to_center)
        distance_to_center /= n_clusters*(n_clusters-1) #average_distance to center
        ratio_of_dist = avg_distance/max_distance_to_center #ratio to multiply with 
        for i in range(n_clusters):
            #vector from center location to all the clusters
            locations[i][0] = (ratio_of_dist+1)*center_x - ratio_of_dist*locations[i][0]
            locations[i][1] = (ratio_of_dist+1)*center_y - ratio_of_dist*locations[i][1]                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

        np.save('locations.npy',np.array(locations,dtype=int))
        np.save('labels.npy', labels)

        np.save(new_path+'locations.npy',np.array(locations,dtype=int))
        np.save(new_path+'labels.npy', labels)
        resp = defaultdict(list)
        cur_children = children[str(json_data['depth']) + '/' + str(json_data['breadth'])]
        cur_parent = parent[str(json_data['depth']) + '/' + str(json_data['breadth'])]
        for idx, i in enumerate(range(n_clusters)):
            representative_img = np.where(labels==i)[0]
            list_of_destinations = []
            for img_number in representative_img:
                list_of_destinations.append(new_path[7:] + str(img_number) +'.png')
            resp[i] = [int(locations[idx][0]), int(locations[idx][1]), list_of_destinations]        
        #locations = locations.tolist()
        with open('public/state.txt', 'w') as f:
            f.write(json.dumps(json_data))
        resp[n_clusters] = ([len(cur_parent),len(cur_children)])
        resp = json.dumps(resp)
        with open(new_path + 'render.txt', 'w') as f:
            f.write(resp)
        resp = make_response(resp)
        return resp

    return jsonify({})

@app.route('/save_user_action', methods=['GET', 'POST'])
def save_user_action():
    #This is the first scatter, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':
        datafromjs = request.form['mydata']
        action = yaml.load(datafromjs)
        with open('public/actions.txt', 'a') as file:
            file.write(str(time.time()) + ',' + str(action) + '\n')
    return jsonify({})

@app.route('/test_area', methods=['GET', 'POST'])
def test_area():
    #This is the first scatter, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':
        datafromjs = request.form['mydata']
        test_image_location = yaml.load(datafromjs)
        app.logger.info('pogchamp')
        app.logger.info(test_image_location)
        test_image_location = 'public/' + test_image_location
        with open('public/actions.txt', 'a') as file:
            file.write(str(time.time()) + ',' + str('test_area') + ',' + str(yaml.load(datafromjs)) + '\n')
        if 'test' in test_image_location:  
            split_point = test_image_location.rfind('_')
            split_point2 = test_image_location.rfind('/')
            id_of_image = int(test_image_location[split_point2+13:split_point])
            app.logger.info('heeeey')
            app.logger.info(id_of_image)
            path_to_image_folder = test_image_location[:split_point2+1]
            cur_strength = int(test_image_location[split_point+1:][:-4])
            app.logger.info(cur_strength)
        else:
            cur_strength = 5
            split_point = test_image_location.rfind('/')
            path_to_image_folder = test_image_location[:split_point+1]
            id_of_image = int(test_image_location[split_point+1:][:-4])
        
        with open('public/state.txt', 'r') as f:
            json_data = json.load(f)
            all_seeds = json_data['seed']
            test_seeds = json_data['test_seed']
        
        with open(path_to_image_folder + 'directions_list.npy', 'rb') as f:
            directions_list = np.load(f)
        directions = pickle.load( open('directions.pkl', "rb" ) )
        selected_filters = pickle.load( open('selected_filters.pkl', "rb" ) )

        direction = directions[0]
        current_direction = directions_list[id_of_image] 
        OG = sty.Generator(G.z_dim,G .c_dim,G.w_dim,G.img_resolution,G.img_channels, synthesis_kwargs=args.G_kwargs.synthesis_kwargs).cuda()
        OG.load_state_dict(G.state_dict(), strict=False)
        OG.eval()
        for layer in json_data['layers']:
            if layer != 4:
                cur_block = getattr(OG.styling, 'b' + str(layer))
                cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
                for idx, affine in enumerate(affines):
                    cur_aff = getattr(cur_block, affine)
                    cur_aff_target = getattr(cur_block_target, 'conv' + str(idx))
                    cur_aff_target = getattr(cur_aff_target,'affine')
                    for weight in weights: # G.synthesis.b128.conv1.affine.weight
                        setattr(cur_aff, weight, getattr(cur_aff_target,weight))
            else:
                cur_block = getattr(OG.styling, 'b' + str(layer))
                cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
                cur_aff = getattr(cur_block, 'affine1')
                cur_aff_target = getattr(cur_block_target, 'conv1')
                cur_aff_target = getattr(cur_aff_target,'affine')
                for weight in weights: # G.synthesis.b128.conv1.affine.weight
                    setattr(cur_aff, weight, getattr(cur_aff_target,weight))

        all_z = np.stack([np.random.RandomState(all_seeds[json_data['x']]).randn(G.z_dim)]) #selected image z
        all_w = OG.mapping(torch.from_numpy(all_z).to(device), None)
        direction_of_image = []
        
        styles = OG.styling(all_w)
        new_style = []
        for idx, selects in enumerate(selected_filters):
            if idx == 0:
                direction_of_image.append((None, styles[idx+1][0][:,selects[1]]))
            elif idx <= len(json_data['layers'])-2:
                direction_of_image.append((styles[idx][1][:,selects[0]], styles[idx+1][0][:,selects[1]]))
            else:
                direction_of_image.append((styles[idx][1][:,selects[0]], None))
        

        directions_list_tmp = []
        
        dir_tmp = direction_of_image[0][1][0]
        for i in range(1,len(json_data['layers'])-1):
            dir_tmp = torch.cat((dir_tmp,direction_of_image[i][0][0],direction_of_image[i][1][0]))
        dir_tmp = torch.cat((dir_tmp,direction_of_image[len(json_data['layers'])-1][0][0]))
        directions_list_tmp.append(dir_tmp.unsqueeze(0))

        direction_of_image = torch.cat(directions_list_tmp)
        direction_of_image = direction_of_image.cpu().detach().numpy()[0]

        #apply
        chsn = direction[0][1][0].shape[0]
        dir_tmp = []
        pc_values = current_direction
        dir_tmp.append((None,torch.tensor(pc_values[:chsn]).float()))
        for i in range(1,len(json_data['layers'])-1):
            chsn0 = direction[i][0][0].shape[0]
            chsn1 = direction[i][1][0].shape[0]
            dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn0]).float(),torch.tensor(pc_values[chsn+chsn0:chsn+chsn0+chsn1]).float()))
            chsn += chsn0 + chsn1
        chsn1 = direction[len(json_data['layers'])-1][0][0].shape[0]
        dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn1]).float(),None))

        new_styles = OG.styling(all_w)
        for idx, selects in enumerate(selected_filters):
            if idx == 0:
                dir1 = dir_tmp[0][1]
                new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda()) 

            elif idx <= len(json_data['layers'])-2:
                if idx == 2 or idx == 3:
                    dir0 = dir_tmp[idx][0]
                else:
                    dir0 = dir_tmp[idx][0]
                new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())

                if idx == 2 or idx == 3:
                    dir1 = dir_tmp[idx][1]
                else:
                    dir1 = dir_tmp[idx][1]
                new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda())


            else:
                dir0 = dir_tmp[idx][0]
                new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())   
        new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
        new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
        save_location = path_to_image_folder +  'testtarget0' + '_' + str(id_of_image) + '_' + str(cur_strength) + '.png'
        np.save(path_to_image_folder +'current_direction' + str(id_of_image) + '.npy', current_direction)
        save_image(new_images_tmp[0], save_location)
        directions_new = ['ganzilla_images/initial_images/generated' + str(json_data['x']) + '.png', save_location[7:]]
        test_image_seeds = test_seeds
        strlistfortests = ['testtarget1', 'testtarget2', 'testtarget3']
        for test_image_id in range(3):
            all_z_tmp = np.stack([np.random.RandomState(test_image_seeds[test_image_id]).randn(G.z_dim)]) #selected image z
            all_w_tmp = OG.mapping(torch.from_numpy(all_z_tmp).to(device), None)
            actual_direction = current_direction - direction_of_image

            direction_of_image_new = []
        
            styles_new = OG.styling(all_w_tmp)
            new_style = []
            for idx, selects in enumerate(selected_filters):
                if idx == 0:
                    direction_of_image_new.append((None, styles_new[idx+1][0][:,selects[1]]))
                elif idx <= len(json_data['layers'])-2:
                    direction_of_image_new.append((styles_new[idx][1][:,selects[0]], styles_new[idx+1][0][:,selects[1]]))
                else:
                    direction_of_image_new.append((styles_new[idx][1][:,selects[0]], None))
            

            directions_list_tmp2 = []
            
            dir_tmp2 = direction_of_image_new[0][1][0]
            for i in range(1,len(json_data['layers'])-1):
                dir_tmp2 = torch.cat((dir_tmp2,direction_of_image_new[i][0][0],direction_of_image_new[i][1][0]))
            dir_tmp2 = torch.cat((dir_tmp2,direction_of_image_new[len(json_data['layers'])-1][0][0]))
            directions_list_tmp2.append(dir_tmp2.unsqueeze(0))

            direction_of_image_new = torch.cat(directions_list_tmp2)
            direction_of_image_new = direction_of_image_new.cpu().detach().numpy()[0]
            pc_values_new = actual_direction + direction_of_image_new

            #apply
            chsn = direction[0][1][0].shape[0]
            dir_tmp = []
            dir_tmp.append((None,torch.tensor(pc_values_new[:chsn]).float()))
            for i in range(1,len(json_data['layers'])-1):
                chsn0 = direction[i][0][0].shape[0]
                chsn1 = direction[i][1][0].shape[0]
                dir_tmp.append((torch.tensor(pc_values_new[chsn:chsn+chsn0]).float(),torch.tensor(pc_values_new[chsn+chsn0:chsn+chsn0+chsn1]).float()))
                chsn += chsn0 + chsn1
            chsn1 = direction[len(json_data['layers'])-1][0][0].shape[0]
            dir_tmp.append((torch.tensor(pc_values_new[chsn:chsn+chsn1]).float(),None))

            new_styles = OG.styling(all_w_tmp)
            for idx, selects in enumerate(selected_filters):
                if idx == 0:
                    dir1 = dir_tmp[0][1]
                    new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda()) 

                elif idx <= len(json_data['layers'])-2:
                    if idx == 2 or idx == 3:
                        dir0 = dir_tmp[idx][0]
                    else:
                        dir0 = dir_tmp[idx][0]
                    new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())

                    if idx == 2 or idx == 3:
                        dir1 = dir_tmp[idx][1]
                    else:
                        dir1 = dir_tmp[idx][1]
                    new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda())


                else:
                    dir0 = dir_tmp[idx][0]
                    new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())   
            new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
            new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
            save_location2 = path_to_image_folder +  'testtarget' + str(test_image_id+1) + '_' + str(id_of_image) + '_' + str(cur_strength) +  '.png'
            save_image(new_images_tmp[0], save_location2)


            #apply
            chsn = direction[0][1][0].shape[0]
            dir_tmp = []
            dir_tmp.append((None,torch.tensor(direction_of_image_new[:chsn]).float()))
            for i in range(1,len(json_data['layers'])-1):
                chsn0 = direction[i][0][0].shape[0]
                chsn1 = direction[i][1][0].shape[0]
                dir_tmp.append((torch.tensor(direction_of_image_new[chsn:chsn+chsn0]).float(),torch.tensor(direction_of_image_new[chsn+chsn0:chsn+chsn0+chsn1]).float()))
                chsn += chsn0 + chsn1
            chsn1 = direction[len(json_data['layers'])-1][0][0].shape[0]
            dir_tmp.append((torch.tensor(direction_of_image_new[chsn:chsn+chsn1]).float(),None))

            new_styles = OG.styling(all_w_tmp)
            
            new_images = OG.synthesis(all_w_tmp, new_styles, noise_mode='const')
            new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
            save_location2_ref = path_to_image_folder +  'testref'  + str(test_image_id+1) + '_' + str(id_of_image) + '_' + str(cur_strength) +  '.png'
            save_image(new_images_tmp[0], save_location2_ref)
            directions_new.append(save_location2_ref[7:])
            directions_new.append(save_location2[7:])
        resp = json.dumps(directions_new)
        resp = make_response(resp)
    return resp

@app.route('/change_strength', methods=['GET', 'POST'])
def change_strength():
    #This is the first scatter, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':
        datafromjs = request.form['mydata']
        strength_direction, test_image_location, selected_img = yaml.load(datafromjs)
        app.logger.info('pogchamp')
        app.logger.info(test_image_location)
        test_image_location = 'public/' + test_image_location
        with open('public/actions.txt', 'a') as file:
            file.write(str(time.time()) + ',' + str('change_strength') + ',' + str(yaml.load(datafromjs)) + '\n')
        if 'test' in test_image_location:
            split_point = test_image_location.rfind('_')
            split_point2 = test_image_location.rfind('/')
            id_of_image = int(test_image_location[split_point2+13:split_point])
            app.logger.info('heeeey')
            app.logger.info(id_of_image)
            path_to_image_folder = test_image_location[:split_point2+1]
            cur_strength = int(test_image_location[split_point+1:][:-4])
            app.logger.info(cur_strength)
        else:
            cur_strength = 5
            split_point = test_image_location.rfind('/')
            path_to_image_folder = test_image_location[:split_point+1]
            id_of_image = int(test_image_location[split_point+1:][:-4])
        
        with open('public/state.txt', 'r') as f:
            json_data = json.load(f)
            all_seeds = json_data['seed']
            test_seeds = json_data['test_seed']
        
        with open(path_to_image_folder + 'directions_list.npy', 'rb') as f:
            directions_list = np.load(f)
        directions = pickle.load( open('directions.pkl', "rb" ) )
        selected_filters = pickle.load( open('selected_filters.pkl', "rb" ) )

        direction = directions[0]
        current_direction = directions_list[id_of_image] 
        OG = sty.Generator(G.z_dim,G .c_dim,G.w_dim,G.img_resolution,G.img_channels, synthesis_kwargs=args.G_kwargs.synthesis_kwargs).cuda()
        OG.load_state_dict(G.state_dict(), strict=False)
        OG.eval()
        for layer in json_data['layers']:
            if layer != 4:
                cur_block = getattr(OG.styling, 'b' + str(layer))
                cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
                for idx, affine in enumerate(affines):
                    cur_aff = getattr(cur_block, affine)
                    cur_aff_target = getattr(cur_block_target, 'conv' + str(idx))
                    cur_aff_target = getattr(cur_aff_target,'affine')
                    for weight in weights: # G.synthesis.b128.conv1.affine.weight
                        setattr(cur_aff, weight, getattr(cur_aff_target,weight))
            else:
                cur_block = getattr(OG.styling, 'b' + str(layer))
                cur_block_target = getattr(OG.synthesis, 'b' + str(layer))
                cur_aff = getattr(cur_block, 'affine1')
                cur_aff_target = getattr(cur_block_target, 'conv1')
                cur_aff_target = getattr(cur_aff_target,'affine')
                for weight in weights: # G.synthesis.b128.conv1.affine.weight
                    setattr(cur_aff, weight, getattr(cur_aff_target,weight))

        save_location = path_to_image_folder +  'testtarget0' + '_' +str(id_of_image) +  '_' + str(strength_direction) + '.png'
        all_z = np.stack([np.random.RandomState(all_seeds[json_data['x']]).randn(G.z_dim)]) #selected image z
        all_w = OG.mapping(torch.from_numpy(all_z).to(device), None)
        direction_of_image = []
        
        styles = OG.styling(all_w)
        new_style = []
        for idx, selects in enumerate(selected_filters):
            if idx == 0:
                direction_of_image.append((None, styles[idx+1][0][:,selects[1]]))
            elif idx <= len(json_data['layers'])-2:
                direction_of_image.append((styles[idx][1][:,selects[0]], styles[idx+1][0][:,selects[1]]))
            else:
                direction_of_image.append((styles[idx][1][:,selects[0]], None))
        

        directions_list_tmp = []
        
        dir_tmp = direction_of_image[0][1][0]
        for i in range(1,len(json_data['layers'])-1):
            dir_tmp = torch.cat((dir_tmp,direction_of_image[i][0][0],direction_of_image[i][1][0]))
        dir_tmp = torch.cat((dir_tmp,direction_of_image[len(json_data['layers'])-1][0][0]))
        directions_list_tmp.append(dir_tmp.unsqueeze(0))

        direction_of_image = torch.cat(directions_list_tmp)
        direction_of_image = direction_of_image.cpu().detach().numpy()[0]

        if selected_img == 0:
            directions_new = ['ganzilla_images/initial_images/generated' + str(json_data['x']) + '.png', save_location[7:],selected_img]
            if not os.path.isfile(save_location):
                app.logger.info('whhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhat')
                app.logger.info(save_location)
                

                app.logger.info(direction_of_image.shape)
                app.logger.info(current_direction.shape)

                
                #apply
                direction = directions[0]
                chsn = direction[0][1][0].shape[0]
                dir_tmp = []
                #scaler = (strength_direction*2+1)
                #if scaler > 0:
                #    final_scaler = (scaler/10 + 0.9)
                #else:
                #    final_scaler = scaler/10 - 0.9
                #pc_values = (current_direction - direction_of_image) * final_scaler + direction_of_image
                pc_values = (current_direction - direction_of_image) * strength_direction/5 + direction_of_image
                dir_tmp.append((None,torch.tensor(pc_values[:chsn]).float()))
                for i in range(1,len(json_data['layers'])-1):
                    chsn0 = direction[i][0][0].shape[0]
                    chsn1 = direction[i][1][0].shape[0]
                    dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn0]).float(),torch.tensor(pc_values[chsn+chsn0:chsn+chsn0+chsn1]).float()))
                    chsn += chsn0 + chsn1
                chsn1 = direction[len(json_data['layers'])-1][0][0].shape[0]
                dir_tmp.append((torch.tensor(pc_values[chsn:chsn+chsn1]).float(),None))

                new_styles = OG.styling(all_w)
                for idx, selects in enumerate(selected_filters):
                    if idx == 0:
                        dir1 = dir_tmp[0][1]
                        new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda()) 

                    elif idx <= len(json_data['layers'])-2:
                        if idx == 2 or idx == 3:
                            dir0 = dir_tmp[idx][0]
                        else:
                            dir0 = dir_tmp[idx][0]
                        new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())

                        if idx == 2 or idx == 3:
                            dir1 = dir_tmp[idx][1]
                        else:
                            dir1 = dir_tmp[idx][1]
                        new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda())


                    else:
                        dir0 = dir_tmp[idx][0]
                        new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())   
                new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
                new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
                save_location = path_to_image_folder +  'testtarget0' + '_' +str(id_of_image) +  '_' + str(strength_direction) + '.png'
                np.save(path_to_image_folder +'current_direction' + str(id_of_image) + '.npy', current_direction)
                save_image(new_images_tmp[0], save_location)

        else:
            test_image_seeds = test_seeds
            for test_image_id in range(3):
                save_location2_ref = path_to_image_folder +  'testref'  + str(test_image_id+1) + '_' + str(id_of_image) + '_' + str(strength_direction) +  '.png'
                save_location2 = path_to_image_folder +  'testtarget' + str(test_image_id+1) + '_' + str(id_of_image) + '_' + str(strength_direction) +  '.png'
                if selected_img == test_image_id+1:
                    directions_new = ([save_location2_ref[7:],save_location2[7:],selected_img])
                    if not os.path.isfile(save_location2_ref):
                        all_z_tmp = np.stack([np.random.RandomState(test_image_seeds[test_image_id]).randn(G.z_dim)]) #selected image z
                        all_w_tmp = OG.mapping(torch.from_numpy(all_z_tmp).to(device), None)

                        direction_of_image_new = []
                    
                        styles_new = OG.styling(all_w_tmp)
                        new_style = []
                        for idx, selects in enumerate(selected_filters):
                            if idx == 0:
                                direction_of_image_new.append((None, styles_new[idx+1][0][:,selects[1]]))
                            elif idx <= len(json_data['layers'])-2:
                                direction_of_image_new.append((styles_new[idx][1][:,selects[0]], styles_new[idx+1][0][:,selects[1]]))
                            else:
                                direction_of_image_new.append((styles_new[idx][1][:,selects[0]], None))
                        

                        directions_list_tmp2 = []
                        
                        dir_tmp2 = direction_of_image_new[0][1][0]
                        for i in range(1,len(json_data['layers'])-1):
                            dir_tmp2 = torch.cat((dir_tmp2,direction_of_image_new[i][0][0],direction_of_image_new[i][1][0]))
                        dir_tmp2 = torch.cat((dir_tmp2,direction_of_image_new[len(json_data['layers'])-1][0][0]))
                        directions_list_tmp2.append(dir_tmp2.unsqueeze(0))

                        direction_of_image_new = torch.cat(directions_list_tmp2)
                        direction_of_image_new = direction_of_image_new.cpu().detach().numpy()[0]
                        pc_values_new = (current_direction - direction_of_image) * strength_direction/5 + direction_of_image_new
                        
                        #apply
                        chsn = direction[0][1][0].shape[0]
                        dir_tmp = []
                        dir_tmp.append((None,torch.tensor(pc_values_new[:chsn]).float()))
                        for i in range(1,len(json_data['layers'])-1):
                            chsn0 = direction[i][0][0].shape[0]
                            chsn1 = direction[i][1][0].shape[0]
                            dir_tmp.append((torch.tensor(pc_values_new[chsn:chsn+chsn0]).float(),torch.tensor(pc_values_new[chsn+chsn0:chsn+chsn0+chsn1]).float()))
                            chsn += chsn0 + chsn1
                        chsn1 = direction[len(json_data['layers'])-1][0][0].shape[0]
                        dir_tmp.append((torch.tensor(pc_values_new[chsn:chsn+chsn1]).float(),None))

                        new_styles = OG.styling(all_w_tmp)
                        for idx, selects in enumerate(selected_filters):
                            if idx == 0:
                                dir1 = dir_tmp[0][1]
                                new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda()) 

                            elif idx <= len(json_data['layers'])-2:
                                if idx == 2 or idx == 3:
                                    dir0 = dir_tmp[idx][0]
                                else:
                                    dir0 = dir_tmp[idx][0]
                                new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())

                                if idx == 2 or idx == 3:
                                    dir1 = dir_tmp[idx][1]
                                else:
                                    dir1 = dir_tmp[idx][1]
                                new_styles[idx+1][0][0][selects[1]] = torch.clone(dir1.cuda())


                            else:
                                dir0 = dir_tmp[idx][0]
                                new_styles[idx][1][0][selects[0]] = torch.clone(dir0.cuda())   
                        new_images = OG.synthesis(all_w, new_styles, noise_mode='const')
                        new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
                        save_image(new_images_tmp[0], save_location2)


                        #apply
                        chsn = direction[0][1][0].shape[0]
                        dir_tmp = []
                        dir_tmp.append((None,torch.tensor(direction_of_image_new[:chsn]).float()))
                        for i in range(1,len(json_data['layers'])-1):
                            chsn0 = direction[i][0][0].shape[0]
                            chsn1 = direction[i][1][0].shape[0]
                            dir_tmp.append((torch.tensor(direction_of_image_new[chsn:chsn+chsn0]).float(),torch.tensor(direction_of_image_new[chsn+chsn0:chsn+chsn0+chsn1]).float()))
                            chsn += chsn0 + chsn1
                        chsn1 = direction[len(json_data['layers'])-1][0][0].shape[0]
                        dir_tmp.append((torch.tensor(direction_of_image_new[chsn:chsn+chsn1]).float(),None))

                        new_styles = OG.styling(all_w_tmp) 
                        new_images = OG.synthesis(all_w_tmp, new_styles, noise_mode='const')
                        new_images_tmp = (new_images[0].clamp(-1, 1) + 1) / 2
                        save_image(new_images_tmp[0], save_location2_ref)
                      
        resp = json.dumps(directions_new)
        resp = make_response(resp)
    return resp

@app.route('/save_direction', methods=['GET', 'POST'])
def save_direction():
    #This is the first scatter, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':
        datafromjs = request.form['mydata']
        current_direction = yaml.load(datafromjs)
        app.logger.info(current_direction)
        with open('public/actions.txt', 'a') as file:
            file.write(str(time.time()) + ',' + 'save_direction' + ',' + str(current_direction) + '\n')

    return jsonify({})

@app.route('/go_back', methods=['GET', 'POST'])
def go_back():
    #This is the first scatter, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':


        with open('public/state.txt', 'r') as f:
            json_data = json.load(f)
            cur_x = json_data['x']
            cur_y = json_data['y']
            cur_depth = json_data['depth']
            cur_breadth = json_data['breadth']

        with open('public/actions.txt', 'a') as file:
            file.write(str(time.time()) + ',' + str('go_back') + ',' + str(cur_x) + ',' + str(cur_y) + ',' + str(cur_depth) + ',' + str(cur_breadth) + '\n')
        path_to_xy = 'public/ganzilla_images/raw_images/' + str(cur_x) + '_' + str(cur_y) + '/'
        with open(path_to_xy + "parent.pkl", "rb") as pkl_handle:
	        parent = pickle.load(pkl_handle)  

        cur_parent = parent[str(cur_depth) + '/' + str(cur_breadth)][0]
        path_to_render = path_to_xy + cur_parent + '/'
        split_point = cur_parent.rfind('/')
        new_depth = cur_parent[:split_point]
        new_breadth = cur_parent[split_point+1:]
        json_data['depth'] = int(new_depth)
        json_data['breadth'] = int(new_breadth)

        with open('public/state.txt', 'w') as f:
            f.write(json.dumps(json_data))

        with open(path_to_render + 'render.txt', 'r') as f:
            render_data = json.load(f)
        resp = make_response(json.dumps(render_data))
    return resp

@app.route('/go_next', methods=['GET', 'POST'])
def go_next():
    #This is the first scatter, find relative filters
    #read the seed from saved location (for now I assume we know the seed)
    if request.method == 'POST':
        datafromjs = request.form['mydata']
        next_index = yaml.load(datafromjs)


        with open('public/state.txt', 'r') as f:
            json_data = json.load(f)
            cur_x = json_data['x']
            cur_y = json_data['y']
            cur_depth = json_data['depth']
            cur_breadth = json_data['breadth']

        with open('public/actions.txt', 'a') as file:
            file.write(str(time.time()) + ',' + str('go_next') + ',' + str(cur_x) + ',' + str(cur_y) + ',' + str(cur_depth) + ',' + str(cur_breadth) + '\n')
        path_to_xy = 'public/ganzilla_images/raw_images/' + str(cur_x) + '_' + str(cur_y) + '/'
        with open(path_to_xy + "children.pkl", "rb") as pkl_handle:
	        children = pickle.load(pkl_handle)

        cur_children = children[str(cur_depth) + '/' + str(cur_breadth)][next_index]
        
        path_to_render = path_to_xy + cur_children + '/'
        split_point = cur_children.rfind('/')
        new_depth = cur_children[:split_point]
        new_breadth = cur_children[split_point+1:]
        json_data['depth'] = int(new_depth)
        json_data['breadth'] = int(new_breadth)

        with open('public/state.txt', 'w') as f:
            f.write(json.dumps(json_data))

        with open(path_to_render + 'render.txt', 'r') as f:
            render_data = json.load(f)
        resp = make_response(json.dumps(render_data))
    return resp
if __name__ == "__main__":
    app.run(debug = True, port=5000)
    
    
