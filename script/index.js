let canvas = document.getElementById("myCanvas");
let paramCanvas = document.getElementById("paramChart");
let global_selected_method = "regression";  // Globally selected method
let global_data = undefined;    // Globally generated data
let global_raw_data = undefined;
let global_continue_flag= true; // Stop button flag
let global_rounding_factor = 3; // Global rounding factor for all calculations
let CLIP = [-10,10];
let global_param_index = 0;
let global_last_weights = undefined;

let funcBinder = {
    alg : function(inp){
        let lst = [];
        for(let i=1;i<=inp["ord"];i++){
            lst.push(round(Math.pow(inp["nx"][0],i),global_rounding_factor));
        }
        return lst;
    },
    sin : function(inp){
        let lst = [];
        for(let i=1;i<=inp["ord"];i++){
            lst.push(round(Math.pow(Math.sin(inp["nx"][0]),i),global_rounding_factor));
        }
        return lst;
    },
    cos : function(inp){
        let lst = [];
        for(let i=1;i<=inp["ord"];i++){
            lst.push(round(Math.pow(Math.cos(inp["nx"][0]),i),global_rounding_factor));
        }
        return lst;
    },
    sincos : function(inp){
        let lst = [];
        for(let i=1;i<=inp["ord"];i++){
            lst.push(round(Math.pow(Math.sin(inp["nx"][0]),i)+Math.pow(Math.cos(inp["nx"][0]),i),global_rounding_factor));
        }
        return lst;
    }
}

document.getElementById("uploadInput").onchange = function(e){
    if(e.target.files.length == 0){
        alert("No file selected !");
        return;
    }
    let reader = new FileReader();
    reader.readAsText(e.target.files[0]);
    reader.onload = function(out){
        processCSVText(out.target.result);
    };
}

let chart = new Chart(canvas,{
    type : "scatter",
    options : {
        scales: {
            x : {
                beginAtZero : true
            },
            y : {
                beginAtZero : true
            }
        }
    }
})

let paramChart = new Chart(paramCanvas,{
    type : "line",
    options : {
        scales : {
            x : {
                beginAtZero : true
            },
            y : {
                beginAtZero : true
            }
        }
    }
})


// Select defaults for loading site
selectModel(document.getElementsByClassName("regression")[0]);
updateLabel(undefined);
updateParamSelector(1);

//Methods and function

function round(number,place){
    return Math.round(number*(10**place)) / (10**place);
}

function range(limit){
    let out = [];
    for(let i=0;i<limit;i++){
        out.push(i);
    }
    return out;
}

function createRandomX(no_points){
    min = -1*Math.round(Math.random()*10);
    max = Math.round(Math.random()*10);;
    let step = (max-min) / no_points;
    let randx = [];
    let ord = +document.getElementById("ordSelect").value;
    for(let i=0;i<no_points;i++){
        let x = 0;
        if(!randx.length){
            x = round(min+step,global_rounding_factor);
        }else{
            x = round(step+randx[i-1][0],global_rounding_factor);
        }
        let x_pow = [];
        for(let i=1;i<=ord;i++){
            x_pow.push(Math.pow(x,i));
        }
        randx.push(x_pow);
    }
    return randx;
}

function createRegression(no_points,noise){
    no_points = no_points ? no_points : 100;
    noise = noise ? noise : 3.5;
    let randx = createRandomX(no_points);
    let ang = Math.PI/2 * Math.random();
    ang *= Math.round(Math.random()) ? 1 : -1;
    let randy = [];
    randx.forEach(x => {
        randy.push(round(Math.tan(ang)*x[0] + noise + round(Math.random()*noise,global_rounding_factor),global_rounding_factor));
    });
    global_data = [randx,randy];
    global_raw_data = JSON.parse(JSON.stringify([randx,randy]));
    return [randx,randy];
}

function getModelProperties(){
    return {
        epochs : +document.getElementsByClassName("form-range epochs")[0].value,
        lr : +document.getElementsByClassName("form-range lr")[0].value,
        batch_size : +document.getElementsByClassName("form-range batch")[0].value,
        order : +document.getElementById("ordSelect").value
    }
}

function alterModelPropertyInput(state){
    document.getElementsByClassName("form-range epochs")[0].disabled = state ? state : false;
    document.getElementsByClassName("form-range lr")[0].disabled = state ? state : false;
    document.getElementsByClassName("form-range batch")[0].disabled = state ? state : false;
}

function sliceData(batch_size){
    if(global_data[0].length <= batch_size){
        return global_data;
    }
    batchwise_data = []
    for(dataset of global_data){
        data = [];
        count = 1;
        batch_data = [];
        for(point of dataset){
            if(!(count%batch_size)){
                data.push(batch_data);
                batch_data = [];
                count++;
                continue;
            }
            batch_data.push(point);
            count++;
        }
        data.push(batch_data);
        batchwise_data.push(data);
    }
    return batchwise_data;
}

function evaluate(x,weights){
    ans = [];
    x.forEach(ele => {
        let y = 0;
        for(let i=0;i<=weights.m.length;i++){
            if(i==weights.m.length){
                y += weights.c;
                continue;
            }
            y += ele[i] * weights.m[i];
        }
        ans.push(round(y,global_rounding_factor));
    })
    return ans;
}

function difference(ar1,ar2){
    let res= [];
    ar1.forEach(x => {
        res.push(round(x-ar2[ar1.indexOf(x)],global_rounding_factor));
    })
    return res;
}

function loss(y_p,y_t){
    ans = [];
    difference(y_t,y_p).forEach(x => {
        ans.push(round(x**2,global_rounding_factor));
    });
    return ans;
}

function average(lst){
    s = 0;
    lst.forEach(x=>{
        s += x;
    });
    return round(s / lst.length,global_rounding_factor);
}

function clip(grad){
    if(grad<CLIP[0]){
        return CLIP[0];
    }
    if(grad>CLIP[1]){
        return CLIP[1];
    }
    return grad;
}

function grad(y_t,y_p,x_p){
    ans_m = [];
    ans_c = [];
    ans_c_set = false;
    diff = difference(y_t,y_p);
    for(let i=0;i<=x_p[0].length;i++){
        let dx_param = [];
        x_p.forEach(x => {
            if(i==x_p[0].length){
                dx_param.push(clip(round(diff[x_p.indexOf(x)],global_rounding_factor)));
            }else{
                dx_param.push(clip(round(diff[x_p.indexOf(x)]*x[i],global_rounding_factor)));
            }
        });
        if(i!=x_p[0].length){
            ans_m.push(average(dx_param));
            continue;
        }
        ans_c.push(average(dx_param));
    }
    return [ans_m,ans_c];
}

function updatePredictionLine(weights){
    let gen = [];
    global_data[0].forEach(x => {
        y = 0;
        for(let i=0;i<=weights.m.length;i++){
            if(i==weights.m.length){
                y += weights.c;
                continue;
            }
            y += weights.m[i] * x[i];
        }
        gen.push(round(y,global_rounding_factor));
    });
    if(chart.data.datasets.length==1){
        chart.data.datasets.push({
            label : "Prediction line",
            data : gen,
            backgroundColor : "red",
            type : "line"     
        });
    }else{
        chart.data.datasets[1] = {
            label : "Prediction line",
            data : gen,
            backgroundColor : "red",
            type : "line"
        };
    }
    chart.update();
}

function shuffle(x_arr,y_arr){
    for (let i = x_arr.length-1; i >= 0; i--) {
        let rand = Math.round(Math.random()*(x_arr.length-1));
        let c1 = x_arr[i];
        let c2 = y_arr[i];

        x_arr[i] = x_arr[rand];
        y_arr[i] = y_arr[rand];
        x_arr[rand] = c1;
        y_arr[rand] = c2;
    }
}

function initWeights(order){
    let w = []
    for(let i=0;i<order;i++){
        w.push(round(Math.random(),global_rounding_factor))
    }
    return w;
}

function runModel(modelProps){
    if(!modelProps){
        alert("Model properties not aquired !");
        return;
    }
    let data_batchwise = sliceData(modelProps["batch_size"]);
    let x_batchwise = data_batchwise[0];
    let y_batchwise = data_batchwise[1];
    let ord = modelProps["order"];

    let weights = {
        m : initWeights(ord),
        c : round(Math.random(),global_rounding_factor),
        update : function(new_m,new_c){
            for(let i=0;i<this.m.length;i++){
                this.m[i] = round(this.m[i] + new_m[i],global_rounding_factor)
            }
            this.c = round(this.c+new_c,global_rounding_factor);
        }
    }
    all_loss = [];
    for(let i=0;i<modelProps["epochs"];i++){
        epoch_lss = [];
        shuffle(x_batchwise,y_batchwise);
        x_batchwise.forEach(x_batch=>{
            shuffle(x_batch,y_batchwise.at(x_batchwise.indexOf(x_batch)));
        });
        if(global_continue_flag){
            for(let j=0;j<x_batchwise.length;j++){
                y_pred = evaluate(x_batchwise[j],weights);
                lss = loss(y_pred,y_batchwise[j]);
                grd = grad(y_batchwise[j],y_pred,x_batchwise[j]);
                grd[0].forEach(ele=>{
                    grd[0][grd[0].indexOf(ele)] = round(ele*(modelProps["lr"]),global_rounding_factor);
                });
                weights.update(grd[0],round(grd[1]*modelProps["lr"],global_rounding_factor));
                epoch_lss.push(average(lss));
            }
        }else{
            stopModel(document.getElementById("stopButton"));
        }
        populateResult([i+1,average(epoch_lss),weights],["Epoch","Loss","Weights"]);
        all_loss.push(average(epoch_lss));
        global_last_weights = weights;
        updatePredictionLine(weights);
        console.log(`Epoch : ${i}   Loss : ${average(epoch_lss)}`);
    }
    updateParamChart(all_loss);
}

function populateResult(props,prop_names){
    let eles = document.getElementsByClassName("status-output");
    eles[0].innerText = `Epoch : ${props[0]}`;
    eles[1].innerText = `Loss : ${props[1]}`;
    eles[2].innerText = `Weigths : ${props[2].m} , ${props[2].c}`;
}

// Events and clicks

function selectionChanged(obj){
    if(obj.id == "targetSelect"){
        return;
    }
    for(let i=0;i<obj.childElementCount;i++){
        if(obj.children[i].value == obj.value){
            if(i > global_data[0][0].length-1){
                global_param_index = -1;
                continue;
            }
            global_param_index = i;
        }
    }
    load_chart(undefined);
    if(global_last_weights){
        updatePredictionLine(global_last_weights);
    }
}

function uploadCSV(obj){
    document.getElementById("uploadInput").click();
}

function processCSVText(data){
    let split = data.split("\n");
    // console.log(split[0][2]);
    if(data.length == 0){
        alert("Empty file !");
        return;
    }
    if(split.length == 0 || split.length < 2){
        alert("Less data !");
        return;
    }
    if(split[0].length<2){
        alert("No parameter received (Only one data column)!");
        return;
    }
    let params = "";
    split.forEach(ele => {
        split[split.indexOf(ele)] = ele.split(",");
    });
    for(let val of split[0]){
        params += val+" ";
    }
    setGlobalData(split);
}

function setGlobalData(data){
    let params = data.splice(0,1)[0];
    let nans = [];
    console.log(params);
    for(let i=0;i<params.length;i++){
        let trial = data[0][i];
        trial = +trial;
        if(isNaN(trial)){
            nans.push(params[i]);
        }
        console.log(trial);
    }

    data.forEach(ele => {
        let newEle = [];
        for(let i=0;i<ele.length;i++){
            let isnan = false;
            for(let nan of nans){
                if(i == params.indexOf(nan)){
                    isnan = true;
                }
            }
            if(!isnan){
                newEle.push(ele[i]);
            }
        }
        data[data.indexOf(ele)] = newEle;
    });

    for(let nan of nans){
        params.splice(params.indexOf(nan),1);
    }

    alert("Enter the target column from given : "+params);
    let res = prompt("Target : ");
    console.log(res);

    let target_ind = 0;
    for(let i=0;i<data[0].length;i++){
        if(params[i] == res){
            target_ind = i;
        }
    }
    console.log(params);
    let target = params.splice(target_ind,1);
    console.log(params);
    console.log(target);
    let x_vals = [];
    let y_vals = [];
    
    data.forEach(ele => {
        let ele_lst = [];
        for(let i=0;i<ele.length;i++){
            if(i!=target_ind){
                ele_lst.push(+ele[i]);
                continue;
            }
            y_vals.push(+ele[i]);
        }
        x_vals.push(ele_lst);
    });
    x_vals.forEach(ele => {
        if(ele.length==0){
            x_vals.splice(x_vals.indexOf(ele),1);
        }
    });
    y_vals.forEach(ele => {
        if(ele == undefined){
            y_vals.splice(y_vals.indexOf(ele),1);
        }
    });
    if(x_vals.length != y_vals.length){
        alert("Assertion error in lengths of params and targets !");
        return;
    }
    global_data = [x_vals,y_vals];
    updateDataModelControls(params,target);
}

function updateDataModelControls(x_params,y_target){
    document.getElementById("oRange").disabled = true;
    document.getElementById("noRange").disabled = true;
    document.getElementById("noise").disabled = true;

    updateParamSelector({
        x_param : x_params,
        y_target : y_target
    });

    load_chart(undefined);
}

//Test method to check the random generation
function load_chart(options){
    let data = undefined;
    if(options){
        if(!global_selected_method){
            alert("global_selected_method undefined !");
            return;
        }
        data = undefined;
        if(global_selected_method=="regression"){
            if(!options){
                data = createRegression(100,4.5);
            }else{
                data = createRegression(options["no_points"],options["noise"]);
            }
        }
        if(!data){
            alert("data is undefined !");
            return;
        }
    }else{
        if(global_param_index==-1){
            data = global_raw_data;  
            global_param_index = 0;
        }else{
            data = global_data;
        }
    }
    chart.data = {
        labels : selectParamColumn(data[0],global_param_index),
        datasets : [
            {
                label : "Regression Points",
                data : data[1],
                backgroundColor : "green"
            }
        ],
    }
    chart.update();
}

function selectParamColumn(data,column_index){
    let param_data = [];
    data.forEach(ele => {
        param_data.push(ele[column_index]);
    });
    return param_data;
}

function updateParamChart(newParams){
    paramChart.data = {
        labels : range(newParams.length),
        datasets : [
            {
                label : "Loss",
                data : newParams,
                backgroundColor : "red",
                borderColor : "red"
            }
        ]
    }
    paramChart.update();
}


//Runned when model is selected
function selectModel(object){
    global_selected_method = object.classList[2];
    object.className = "btn btn-success "+global_selected_method+" mdl_btn";
    let objs = Array.from(document.getElementsByClassName("mdl_btn"));
    objs.splice(objs.indexOf(object),1);
    objs.forEach(x => {
        if(x.classList.length > 3){
            x.className = "btn btn-secondary "+x.classList[2]+" mdl_btn";
        }
    })
    document.getElementsByClassName("status-output")[0].innerText = "";
    updateChart();
}

//Runned when properties label is updated
function updateLabel(object){
    if(!object){
        let objs = Array.from(document.getElementsByClassName("form-range"));
        objs.forEach(x => {
            document.getElementsByClassName("control-label-view "+x.classList[1])[0].innerText = x.value;
        });
        return;
    }
    let property = object.classList[1];
    document.getElementsByClassName("control-label-view "+property)[0].innerText = object.value;
}

function updateGlobalDataOrder(obj){
    let order = +obj.value;
    global_raw_data[0].forEach(x => {
        global_data[0][global_raw_data[0].indexOf(x)] = funcBinder[document.getElementById("funcSelect").value]({ord : order, nx : x});
    });
    updateParamSelector(order);
    load_chart(undefined);
}

function createOption(value,innerText,theParent){
    let opt = document.createElement("option");
    opt.value = value;
    opt.innerText = innerText;
    theParent.appendChild(opt);
}

function updateParamSelector(order){
    let paramSelector = document.getElementById("paramSelect");
    let targetSelector = document.getElementById("targetSelect");

    while(paramSelector.firstChild){
        paramSelector.removeChild(paramSelector.firstChild);
    }
    if(Object.keys(order).length){
        for(let i=0;i<order["x_param"].length;i++){
            createOption(order["x_param"][i],order["x_param"][i],paramSelector);
        }
    }else{
        for(let i=1;i<=order;i++){
            createOption("x^"+i,"x^"+i,paramSelector);
        }
        if(document.getElementById("funcSelect").value != "alg"){
            createOption("x","x",paramSelector);
        }
    }
    targetSelector.removeChild(targetSelector.firstChild);
    if(Object.keys(order).length){
        createOption(order["y_target"],order["y_target"],targetSelector);
    }else{
        createOption("y","y",targetSelector);
    }
}

// Update Global Rounding Factor
function updateGlobalRoundingFactor(object){
    global_rounding_factor = +object.value;
}

//Update chart on change of range
function updateChart(object){
    let noise;
    let no_points;

    let objs = Array.from(document.getElementsByClassName("form-range"));
    objs.forEach(x => {
        if(x.classList[1]=="noise"){
            noise = +x.value;
        }
        if(x.classList[1]=="range"){
            no_points = +x.value;
        }
    });
    if(noise && no_points){
        load_chart({
            no_points : no_points,
            noise : noise
        })
    }
}

//Start Model Button Onclick
function startModel(object){
    global_continue_flag = true;
    object.disabled = true;
    document.getElementById("stopButton").disabled = false;

    model_props = getModelProperties();
    alterModelPropertyInput(true);
    runModel(model_props);

    object.disabled = false;
    document.getElementById("stopButton").disabled = true;
    alterModelPropertyInput();
}

//Stop Model Button onclick
function stopModel(object){
    global_continue_flag = false;
    object.disabled = true;
    document.getElementById("startButton").disabled = false;
    alterModelPropertyInput();
}