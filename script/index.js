let canvas = document.getElementById("myCanvas");
let paramCanvas = document.getElementById("paramChart");
let global_selected_method = "regression";  // Globally selected method
let global_data = undefined;    // Globally generated data
let global_continue_flag= true; // Stop button flag
let global_rounding_factor = 3; // Global rounding factor for all calculations
let CLIP = [-10,10];

// Test variables
let obj = undefined;

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
    for(let i=0;i<no_points;i++){
        if(!randx.length){
            randx.push(round(min+step,global_rounding_factor));
            continue;
        }
        randx.push(round(step+randx[i-1],global_rounding_factor));
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
        randy.push(round(Math.tan(ang)*x + noise + round(Math.random()*noise,global_rounding_factor),global_rounding_factor));
    });
    global_data = [randx,randy];
    return [randx,randy];
}

function getModelProperties(){
    return {
        epochs : +document.getElementsByClassName("form-range epochs")[0].value,
        lr : +document.getElementsByClassName("form-range lr")[0].value,
        batch_size : +document.getElementsByClassName("form-range batch")[0].value,
        order : +document.getElementsByClassName("form-range order")[0].value
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
        data = []
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
        for(let i=weights.m.length;i>=0;i--){
            if(i==0){
                y += weights.c;
                continue;
            }
            y += Math.pow(ele,i) * weights.m[weights.m.length-i];
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

function grad(y_t,y_p,x_p,ord){
    ans_m = [];
    ans_c = [];
    ans_c_set = false;
    diff = difference(y_t,y_p);
    for(let i=ord;i>=1;i--){
        gd = [];
        x_p.forEach(ele=>{
            gd.push(clip(round(diff[x_p.indexOf(ele)]*Math.pow(ele,i),global_rounding_factor)));
            if(!ans_c_set){
                ans_c.push(clip(round(diff[x_p.indexOf(ele)],global_rounding_factor)));
                ans_c_set = true;
            }
        })
        ans_m.push(gd);
    }
    avg_gd = []
    ans_m.forEach(ele=>{
        avg_gd.push(average(ele));
    })
    return [avg_gd,average(ans_c)];
}

function updatePredictionLine(weights){
    let gen = [];
    global_data[0].forEach(x => {
        y = 0;
        for(let i=weights.m.length;i>=0;i--){
            if(i==0){
                y += weights.c;
                continue;
            }
            y += weights.m[weights.m.length-i] * Math.pow(x,i);
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

<<<<<<< HEAD
function initWeights(order){
    let w = []
    for(let i=0;i<order;i++){
        w.push(round(Math.random(),global_rounding_factor))
    }
    return w;
}

=======
>>>>>>> 11a6af8ec591a0dc1b5717c24dabd462b8382259
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
                grd = grad(y_batchwise[j],y_pred,x_batchwise[j],ord);
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

//Test method to check the random generation
function load_chart(options){
    if(!global_selected_method){
        alert("global_selected_method undefined !");
        return;
    }
    let data = undefined;
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
    chart.data = {
        labels : data[0],
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

function updateParamChart(newParams){
<<<<<<< HEAD
=======
    // console.log(newParams);
>>>>>>> 11a6af8ec591a0dc1b5717c24dabd462b8382259
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