
var selected_x = 0
var flask_url = "http://localhost:5000/"
var ui_url = "http://localhost:3000/"
for (let step = 1; step < 11; step++) {
    $('.select' + step).click(function(){
        $('.selected').removeClass('selected');
        $(this).addClass('selected');
        selected_x = step-1;
        console.log(selected_x);
    });
}

function update_x(input){
    var jqXHR = $.ajax({
        type: "POST",
        url: flask_url + "/update_x",
        data: { mydata: JSON.stringify(input) },
        success: gotoganzilla,
    });
  }
function gotoganzilla(response) {
    window.location.href = ui_url + "ganzilla";
}
$('.selectbutton').click(function (e) {
    update_x(selected_x);
    
});

function selector_opens(input){
    var jqXHR = $.ajax({
        type: "POST",
        url: flask_url + "selector_opens",
        data: { mydata: JSON.stringify(input) },
        success: first_executed,
    });
  } 

function first_executed(response) {
    resp = JSON.parse(response);
    for (let step = 0; step < resp.length; step = step + 1) {
        $('.select' + (step+1)).attr('src', resp[step])
    }
    for (let step = 10; step > resp.length; step = step - 1) {
        $('.select' + step).hide();
    }
}
function codeAddress() {
    selector_opens();
  }
  window.onload = codeAddress;

