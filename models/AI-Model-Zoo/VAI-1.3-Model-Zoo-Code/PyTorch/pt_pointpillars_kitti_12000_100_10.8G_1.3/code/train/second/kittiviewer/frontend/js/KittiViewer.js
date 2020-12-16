var KittiViewer = function (pointCloud, logger, imageCanvas) {
    this.rootPath = "/path/to/kitti";
    this.infoPath = "/path/to/infos.pkl";
    this.detPath = "/path/to/results.pkl";
    this.backend = "http://127.0.0.1:7000";
    this.checkpointPath = "/path/to/tckpt";
    this.configPath = "/path/to/config";
    this.drawDet = false;
    this.imageIndexes = [];
    this.imageIndex = 1;
    this.gtBoxes = [];
    this.dtBoxes = [];
    this.gtBboxes = [];
    this.dtBboxes = [];
    this.pointCloud = pointCloud;
    this.maxPoints = 150000;
    this.pointVertices = new Float32Array(this.maxPoints * 3);
    this.gtBoxColor = "#00ff00";
    this.dtBoxColor = "#ff0000";
    this.gtLabelColor = "#7fff00";
    this.dtLabelColor = "#ff7f00";
    this.logger = logger;
    this.imageCanvas = imageCanvas;
    this.image = '';
};

KittiViewer.prototype = {
    readCookies : function(){
        if (CookiesKitti.get("kittiviewer_backend")){
            this.backend = CookiesKitti.get("kittiviewer_backend");
        }
        if (CookiesKitti.get("kittiviewer_rootPath")){
            this.rootPath = CookiesKitti.get("kittiviewer_rootPath");
        }
        if (CookiesKitti.get("kittiviewer_detPath")){
            this.detPath = CookiesKitti.get("kittiviewer_detPath");
        }
        if (CookiesKitti.get("kittiviewer_checkpointPath")){
            this.checkpointPath = CookiesKitti.get("kittiviewer_checkpointPath");
        }
        if (CookiesKitti.get("kittiviewer_configPath")){
            this.configPath = CookiesKitti.get("kittiviewer_configPath");
        }
        if (CookiesKitti.get("kittiviewer_infoPath")){
            this.infoPath = CookiesKitti.get("kittiviewer_infoPath");
        }
    },
    load: function () {
        let self = this;
        let data = {};
        data["root_path"] = this.rootPath;
        data["info_path"] = this.infoPath;
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/readinfo',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("load kitti info fail, please check your backend!");
                console.log("load kitti info fail, please check your backend!");
            },
            success: function (response) {
                let result = response["results"][0];
                self.imageIndexes = [];
                for (var i = 0; i < result["image_indexes"].length; ++i)
                    self.imageIndexes.push(result["image_indexes"][i]);
                self.logger.message("load kitti info success!");
            }
        });
    },
    addhttp: function (url) {
        if (!/^https?:\/\//i.test(url)) {
            url = 'http://' + url;
        }
        return url
    },

    loadDet: function () {
        let self = this;
        let data = {};
        data["det_path"] = this.detPath;
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/read_detection',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("load kitti det fail!");
                console.log("load kitti det fail!");
            },
            success: function (response) {
                self.logger.message("load kitti det success!");
            }
        });
    },
    buildNet: function( ){
        let self = this;
        let data = {};
        data["checkpoint_path"] = this.checkpointPath;
        data["config_path"] = this.configPath;
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/build_network',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("build kitti det fail!");
                console.log("build kitti det fail!");
            },
            success: function (response) {
                self.logger.message("build kitti det success!");
            }
        });
    },
    inference: function( ){
        let self = this;
        let data = {"image_idx": self.imageIndex};
        return $.ajax({
            url: this.addhttp(this.backend) + '/api/inference_by_idx',
            method: 'POST',
            contentType: "application/json",
            data: JSON.stringify(data),
            error: function (jqXHR, exception) {
                self.logger.error("inference fail!");
                console.log("inference fail!");
            },
            success: function (response) {
                response = response["results"][0];
                var locs = response["dt_locs"];
                var dims = response["dt_dims"];
                var rots = response["dt_rots"];
                var scores = response["dt_scores"];
                self.dtBboxes = response["dt_bbox"];
                for (var i = 0; i < self.dtBoxes.length; ++i) {
                    for (var j = self.dtBoxes[i].children.length - 1; j >= 0; j--) {
                        self.dtBoxes[i].remove(self.dtBoxes[i].children[j]);
                    }
                    scene.remove(self.dtBoxes[i]);
                    self.dtBoxes[i].geometry.dispose();
                    self.dtBoxes[i].material.dispose();
                }
                let label_with_score = [];
                for (var i = 0; i < locs.length; ++i) {
                    label_with_score.push("score=" + scores[i].toFixed(2).toString());
                }
                
                self.dtBoxes = boxEdgeWithLabel(dims, locs, rots, 2, self.dtBoxColor,
                    label_with_score, self.dtLabelColor);
                for (var i = 0; i < self.dtBoxes.length; ++i) {
                    scene.add(self.dtBoxes[i]);
                }
                self.drawImage();
            }
        });
    },
    plot: function () {
        return this._plot(this.imageIndex);
    },
    next: function () {
        for (var i = 0; i < this.imageIndexes.length; ++i) {
            if (this.imageIndexes[i] == this.imageIndex) {
                if (i < this.imageIndexes.length) {
                    this.imageIndex = this.imageIndexes[i + 1];
                    return this.plot();
                }
            }
        }
    },
    prev: function () {
        for (var i = 0; i < this.imageIndexes.length; ++i) {
            if (this.imageIndexes[i] == this.imageIndex) {
                if (i > 0) {
                    this.imageIndex = this.imageIndexes[i - 1];
                    return this.plot();
                }
            }
        }
    },
    clear: function(){
        for (var i = 0; i < this.gtBoxes.length; ++i) {
            for (var j = this.gtBoxes[i].children.length - 1; j >= 0; j--) {
                this.gtBoxes[i].remove(this.gtBoxes[i].children[j]);
            }
            scene.remove(this.gtBoxes[i]);
            this.gtBoxes[i].geometry.dispose();
            this.gtBoxes[i].material.dispose();
        }
        this.gtBoxes = [];
        for (var i = 0; i < this.dtBoxes.length; ++i) {
            for (var j = this.dtBoxes[i].children.length - 1; j >= 0; j--) {
                this.dtBoxes[i].remove(this.dtBoxes[i].children[j]);
            }
            scene.remove(this.dtBoxes[i]);
            this.dtBoxes[i].geometry.dispose();
            this.dtBoxes[i].material.dispose();
        }
        this.dtBoxes = [];
        this.gtBboxes = [];
        this.dtBboxes = [];
        // this.image = '';
    },
    _plot: function (image_idx) {
        console.log(this.imageIndexes.length);
        if (this.imageIndexes.length != 0 && this.imageIndexes.includes(image_idx)) {
            let data = {};
            data["image_idx"] = image_idx;
            data["with_det"] = this.drawDet;
            let self = this;
            var ajax1 = $.ajax({
                url: this.addhttp(this.backend) + '/api/get_pointcloud',
                method: 'POST',
                contentType: "application/json",
                data: JSON.stringify(data),
                error: function (jqXHR, exception) {
                    self.logger.error("get point cloud fail!!");
                    console.log("get point cloud fail!!");
                },
                success: function (response) {
                    self.clear();
                    response = response["results"][0];
                    var points_buf = str2buffer(atob(response["pointcloud"]));
                    var points = new Float32Array(points_buf);
                    var locs = response["locs"];
                    var dims = response["dims"];

                    var rots = response["rots"];
                    var labels = response["labels"];
                    self.gtBboxes = response["bbox"];
                    self.gtBoxes = boxEdgeWithLabel(dims, locs, rots, 2,
                        self.gtBoxColor, labels,
                        self.gtLabelColor);
                    // var boxes = boxEdge(dims, locs, rots, 2, "rgb(0, 255, 0)");
                    for (var i = 0; i < self.gtBoxes.length; ++i) {
                        scene.add(self.gtBoxes[i]);
                    }
                    if (self.drawDet && response.hasOwnProperty("dt_dims")) {

                        var locs = response["dt_locs"];
                        var dims = response["dt_dims"];
                        var rots = response["dt_rots"];
                        var scores = response["dt_scores"];
                        self.dtBboxes = response["dt_bbox"];
                        let label_with_score = [];
                        for (var i = 0; i < locs.length; ++i) {
                            label_with_score.push("score=" + scores[i].toFixed(2).toString());
                        }
                        
                        self.dtBoxes = boxEdgeWithLabel(dims, locs, rots, 2, self.dtBoxColor,
                            label_with_score, self.dtLabelColor);
                        for (var i = 0; i < self.dtBoxes.length; ++i) {
                            scene.add(self.dtBoxes[i]);
                        }
                    }
                    for (var i = 0; i < Math.min(points.length / 4, self.maxPoints); i++) {
                        self.pointCloud.geometry.attributes.position.array[i * 3] = points[
                            i * 4];
                        self.pointCloud.geometry.attributes.position.array[i * 3 + 1] =
                            points[i * 4 +
                                1];
                        self.pointCloud.geometry.attributes.position.array[i * 3 + 2] =
                            points[i * 4 +
                                2];
                    }
                    self.pointCloud.geometry.setDrawRange(0, Math.min(points.length / 4,
                        self.maxPoints));
                    self.pointCloud.geometry.attributes.position.needsUpdate = true;
                    self.pointCloud.geometry.computeBoundingSphere();
                }
            });
            var ajax2 = $.ajax({
                url: this.addhttp(this.backend) + '/api/get_image',
                method: 'POST',
                contentType: "application/json",
                data: JSON.stringify(data),
                error: function (jqXHR, exception) {
                    self.logger.error("get image fail!!");
                    console.log("get image fail!!");
                },
                success: function (response) {
                    response = response["results"][0];
                    self.image = response["image_b64"];
                }
            });
            $.when(ajax1, ajax2).done(function(){
                // draw image, bbox
                self.drawImage();
            });
        } else {
            if (this.imageIndexes.length == 0){
                this.logger.error("image indexes isn't load, please click load button!");
                console.log("image indexes isn't load, please click load button!!");
            }else{
                this.logger.error("out of range!");
                console.log("out of range!");
            }
        }
    },
    drawImage : function(){
        if (this.image === ''){
            console.log("??????");
            return;
        }
        let self = this;
        var image = new Image();
        // image.src = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAoAAAAKCAYAAACNMs+9AAAAFUlEQVR42mNk+M9Qz0AEYBxVSF+FAAhKDveksOjmAAAAAElFTkSuQmCC";
        // console.log(response["image_b64"]);
        image.onload = function() {
            let aspect = image.width / image.height;
            let w = self.imageCanvas.width;
            self.imageCanvas.height = w / aspect;
            let h = self.imageCanvas.height;
            let ctx = self.imageCanvas.getContext("2d");
            console.log("draw image");
            ctx.drawImage(image, 0, 0, w, h);
            let x1, y1, x2, y2;
            for (var i = 0; i < self.gtBboxes.length; ++i){
                ctx.beginPath();
                x1 = self.gtBboxes[i][0] * w;
                y1 = self.gtBboxes[i][1] * h;
                x2 = self.gtBboxes[i][2] * w;
                y2 = self.gtBboxes[i][3] * h;
                ctx.rect(x1, y1, x2 - x1, y2 - y1);
                ctx.lineWidth = 1;
                ctx.strokeStyle = "green";
                ctx.stroke();    
            }
            for (var i = 0; i < self.dtBboxes.length; ++i){
                ctx.beginPath();
                x1 = self.dtBboxes[i][0] * w;
                y1 = self.dtBboxes[i][1] * h;
                x2 = self.dtBboxes[i][2] * w;
                y2 = self.dtBboxes[i][3] * h;
                ctx.rect(x1, y1, x2 - x1, y2 - y1);
                ctx.lineWidth = 1;
                ctx.strokeStyle = "blue";
                ctx.stroke();    
            }
        };
        image.src = this.image;

    }
}