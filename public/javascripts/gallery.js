var resp;
var cur_cluster;
var cur_test;
var cur_direction;
var chosen_history = [];
var image_set = [];
var flask_url = "http://localhost:5000/"
function save_number_cluster(input){
  var jqXHR = $.ajax({
      type: "POST",
      url: flask_url + "number_cluster",
      data: { mydata: JSON.stringify(input) },
      success: first_executed,
  });
}

function change_strength_direction_test(input){
  var jqXHR = $.ajax({
      type: "POST",
      url: flask_url + "change_strength",
      data: { mydata: JSON.stringify(input) },
      success: location_of_individual_image,
  });
} 

function number_cluster(response) {
  // do something with the response
  console.log(JSON.parse(response));
}

$( "#slider-range-min" ).slider({
  stop: function( event, ui ) {
    save_number_cluster(ui.value);
  }
});

$( "#slider-range-test1" ).slider({
  stop: function( event, ui ) {
    change_strength_direction_test([ui.value,cur_test,0]);
    cur_direction[8] = ui.value
  }
});

$( "#slider-range-test2" ).slider({
  stop: function( event, ui ) {
    change_strength_direction_test([ui.value,cur_test,1]);
    cur_direction[9] = ui.value
  }
});

$( "#slider-range-test3" ).slider({
  stop: function( event, ui ) {
    change_strength_direction_test([ui.value,cur_test,2]);
    cur_direction[10] = ui.value
  }
});

$( "#slider-range-test4" ).slider({
  stop: function( event, ui ) {
    change_strength_direction_test([ui.value,cur_test,3]);
    cur_direction[11] = ui.value
  }
});

function location_of_individual_image(response) {
  cur_response_tmp = JSON.parse(response)
  
  $('.test_ref'+(cur_response_tmp[2]+1)).attr('src', cur_response_tmp[0]);
  $('.test_target'+(cur_response_tmp[2]+1)).attr('src', cur_response_tmp[1]);
  $('.test_ref'+(cur_response_tmp[2]+1)).show();
  $('.test_target'+(cur_response_tmp[2]+1)).show();
  console.log(cur_direction);
  cur_direction[cur_response_tmp[2]*2] = cur_response_tmp[0]
  cur_direction[cur_response_tmp[2]*2+1] = cur_response_tmp[1]
  console.log(cur_direction);
}

function ganzilla_opens(input){
  var jqXHR = $.ajax({
      type: "POST",
      url: flask_url + "ganzilla_opens",
      data: { mydata: JSON.stringify(input) },
      success: first_executed,
  });
} 

function first_executed(response) {
  // do something with the response
  console.log(JSON.parse(response));
  for (let step = 1; step < 11; step = step + 1) {
    $('.gather_c' + step).hide();
  }
  resp = JSON.parse(response);
  
  $.getJSON( "state.txt", function( data ) {
    n_cluster = data['cluster']
    console.log(Object.keys(resp).length);
    for (let step = 0; step < Object.keys(resp).length-1; step++) {
      cur_cluster = resp[step];
      console.log(cur_cluster);
      $('.gather_c' + (step+1)).attr('src', cur_cluster[2][0]);
      $('.gather_c' + (step+1)).css({'left': cur_cluster[0]+'px', 'top': cur_cluster[1]+'px'});
      $('.gather_c' + (step+1)).show()
    }
    var cur_img_idx = 0;
    image_set = []
    for (let i = 0; i < Object.keys(resp).length-1; i++) {      
      cur_cluster = resp[i]
      for (let j = 0; j < cur_cluster[2].length; j++) {
        $('.cex'+(j+1+cur_img_idx)).attr('src', cur_cluster[2][j]);
        $('.cex'+(j+1+cur_img_idx)).show();
        image_set.push(cur_cluster[2][j])
      }
      cur_img_idx += cur_cluster[2].length
      
    }
    
    button_counts = resp[Object.keys(resp).length-1]
    prev_count = button_counts[0]
    next_count = button_counts[1]
    console.log(prev_count);
    console.log(next_count);
    $('.backbutton').hide();
    for (let step = 0; step < 5; step++) {
      $('.nextbutton' + (step+1)).hide();
    }
    if(prev_count == 1){
      $('.backbutton').show();
    }
    for (let step = 0; step < next_count; step++) {
      $('.nextbutton' + (step+1)).show();
    }
  });
}

function test_area(input){
  var jqXHR = $.ajax({
      type: "POST",
      url: flask_url + "test_area",
      data: { mydata: JSON.stringify(input) },
      success: location_of_image,
  });
} 

function location_of_image(response) {
  $('#amount-test1').show()
  $('#amount-test2').show()
  $('#amount-test3').show()
  $('#amount-test4').show()
  $('#slider-range-test1').show()
  $('#slider-range-test2').show()
  $('#slider-range-test3').show()
  $('#slider-range-test4').show()
  $('#strength-text').show()
  $('.savebutton').show()

  cur_response_tmp = JSON.parse(response)
  console.log(cur_response_tmp);
  $('.test_ref1').attr('src', cur_response_tmp[0]);
  $('.test_target1').attr('src', cur_response_tmp[1]);
  $('.test_ref2').attr('src', cur_response_tmp[2]);
  $('.test_target2').attr('src', cur_response_tmp[3]);
  $('.test_ref3').attr('src', cur_response_tmp[4]);
  $('.test_target3').attr('src', cur_response_tmp[5]);
  $('.test_ref4').attr('src', cur_response_tmp[6]);
  $('.test_target4').attr('src', cur_response_tmp[7]);
  $('.test_ref1').show();
  $('.test_target1').show();
  $('.test_ref2').show();
  $('.test_target2').show();
  $('.test_ref3').show();
  $('.test_target3').show();
  $('.test_ref4').show();
  $('.test_target4').show();  
  cur_direction = cur_response_tmp;
  cur_direction.push(5);
  cur_direction.push(5);
  cur_direction.push(5);
  cur_direction.push(5);
}

function codeAddress() {
  ganzilla_opens();
}
window.onload = codeAddress;

function save_user_action(input) {
  var jqXHR = $.ajax({
    type: "POST",
    url: flask_url + "save_user_action",
    data: { mydata: JSON.stringify(input) },
    success: its_done,
  });  
}

function get_more_images(input) {
  var jqXHR = $.ajax({
    type: "POST",
    url: flask_url + "get_more_images",
    data: { mydata: JSON.stringify(input) },
    success: callbackFunc,
  });  
}

function its_done(respone) {

}

for (let step = 1; step < 11; step++) {
  $('.gather_c' + step).click(function(){
      save_user_action(['gather_c_clicked',step])
      $(this).toggleClass('selected');
      $('.highlighted').removeClass('highlighted');
      for (let k = 1; k < 61; k = k + 1) {
        $('.cex' + k).hide();
      }
      var selected = []
      for (let k = 1; k < 11; k = k + 1) {
        selected.push($('.gather_c' + k).hasClass('selected'))
      }
      for (let i =1; i < 61; i++) {
        $('.cex'+i).attr('src', '')
      }
      var cur_img_idx = 0;
      image_set = [];
      for (let i = 0; i < selected.length; i++) {
        if (selected[i]) {
          cur_cluster = resp[i]
          for (let j = 0; j < cur_cluster[2].length; j++) {
            $('.cex'+(j+1+cur_img_idx)).attr('src', cur_cluster[2][j]);
            $('.cex'+(j+1+cur_img_idx)).show();
            image_set.push(cur_cluster[2][j])
          }
          cur_img_idx += cur_cluster[2].length
        }
      }
      var cur_img_idx = 0;
      if(selected.filter(Boolean).length == 0){
        for (let i = 0; i < Object.keys(resp).length-1; i++) {      
          cur_cluster = resp[i]
          for (let j = 0; j < cur_cluster[2].length; j++) {
            $('.cex'+(j+1+cur_img_idx)).attr('src', cur_cluster[2][j]);
            $('.cex'+(j+1+cur_img_idx)).show();
            image_set.push(cur_cluster[2][j])
          }
          cur_img_idx += cur_cluster[2].length
          
        }
      }

      //console.log(selected)
      //if(selected.some(Boolean)){
      //  $('.gallerybutton').text('Scatter');
      //}
      //else {
      //  $('.gallerybutton').text('Get More Images');
      //}

  });
}
$('#amount-test1').hide()
$('#amount-test2').hide()
$('#amount-test3').hide()
$('#amount-test4').hide()
$('#slider-range-test1').hide()
$('#slider-range-test2').hide()
$('#slider-range-test3').hide()
$('#slider-range-test4').hide()
$('#strength-text').hide()
$('.savebutton').hide()
for (let step = 1; step < 61; step++) {
  $('.cex' + step).click(function(){
    save_user_action(['cex_clicked',step])
    $('.highlighted').removeClass('highlighted');
    $(this).addClass('highlighted');
    test_area(image_set[step-1]);
    $( "#slider-range-test1" ).slider( "value", 5)
    $( "#amount-test1" ).val(5)
    $( "#slider-range-test2" ).slider( "value", 5)
    $( "#amount-test2" ).val(5)
    $( "#slider-range-test3" ).slider( "value", 5)
    $( "#amount-test3" ).val(5)
    $( "#slider-range-test4" ).slider( "value", 5)
    $( "#amount-test4" ).val(5)
    cur_test = image_set[step-1]
  });
}

for (let step = 1; step < 7; step++) {
  $('.chosen' + step).click(function(){

    save_user_action(['chosen_clicked',step])
    $('.highlighted_chosen').removeClass('highlighted_chosen');
    $(this).addClass('highlighted_chosen');

    cur_direction = JSON.parse(JSON.stringify(chosen_history[step-1]));
    $('.test_ref1').attr('src', cur_direction[0]);
    $('.test_target1').attr('src', cur_direction[1]);
    $('.test_ref2').attr('src', cur_direction[2]);
    $('.test_target2').attr('src', cur_direction[3]);
    $('.test_ref3').attr('src', cur_direction[4]);
    $('.test_target3').attr('src', cur_direction[5]);
    $('.test_ref4').attr('src', cur_direction[6]);
    $('.test_target4').attr('src', cur_direction[7]);
    $('.test_ref1').show();
    $('.test_target1').show();
    $('.test_ref2').show();
    $('.test_target2').show();
    $('.test_ref3').show();
    $('.test_target3').show();
    $('.test_ref4').show();
    $('.test_target4').show(); 
    $( "#slider-range-test1" ).slider( "value", cur_direction[8])
    $( "#amount-test1" ).val(cur_direction[8])
    $( "#slider-range-test2" ).slider( "value", cur_direction[9])
    $( "#amount-test2" ).val(cur_direction[9])
    $( "#slider-range-test3" ).slider( "value", cur_direction[10])
    $( "#amount-test3" ).val(cur_direction[10])
    $( "#slider-range-test4" ).slider( "value", cur_direction[11])
    $( "#amount-test4" ).val(cur_direction[11])
    cur_test = cur_direction[1]
  });
}

for (let step = 0; step < 6; step = step + 1) {
  $('.chosen' + (step+1)).hide();
}

for (let step = 1; step < 11; step = step + 1) {
    $('.gather_c' + step).hide();
}

for (let step = 1; step < 61; step = step + 1) {
  $('.cex' + step).hide();
}

for (let step = 1; step < 5; step = step + 1) {
  if(step != 1){}
  $('.test_ref' + step).hide();
  $('.test_target' + step).hide();
}

for (let step = 1; step < 11; step = step + 1) {
  $(function() {
    $( ".gather_c" + step).draggable();
  });
}

for (let step = 0; step < 5; step++) {
  $('.nextbutton' + (step+1)).hide();
}
$('.backbutton').hide();
var number_of_clicks = 0;
var get_more = 0;

function runPyScript(input){
  console.log('das');
  var jqXHR = $.ajax({
      type: "POST",
      url: flask_url + "testis",
      data: { mydata: JSON.stringify(input) },
      success: callbackFunc,
  });
} 

function callbackFunc(response) {
  // do something with the response


  console.log('sdadasda');
  console.log(response);
  for (let step = 1; step < 61; step = step + 1) {
    $('.cex' + step).hide();
  }
  for (let step = 1; step < 11; step = step + 1) {
    $('.gather_c' + step).hide();
  }
  resp = JSON.parse(response);
  $('.selected').removeClass('selected');
  $.getJSON( "state.txt", function( data ) {
    n_cluster = data['cluster']
    for (let step = 0; step < Object.keys(resp).length-1; step++) {
      cur_cluster = resp[step];
      console.log(cur_cluster[2][0]);
      $('.gather_c' + (step+1)).attr('src', cur_cluster[2][0]);
      $('.gather_c' + (step+1)).css({'left': cur_cluster[0]+'px', 'top': cur_cluster[1]+'px'});
      $('.gather_c' + (step+1)).show()
    }
    button_counts = resp[Object.keys(resp).length-1]
    prev_count = button_counts[0]
    next_count = button_counts[1]
    console.log(prev_count);
    console.log(next_count);
    $('.backbutton').hide();
    for (let step = 0; step < 5; step++) {
      $('.nextbutton' + (step+1)).hide();
    }
    if(prev_count == 1){
      $('.backbutton').show();
    }
    for (let step = 0; step < next_count; step++) {
      $('.nextbutton' + (step+1)).show();
    }    
  });
  var selected = []
  for (let k = 1; k < 11; k = k + 1) {
    selected.push($('.gather_c' + k).hasClass('selected'))
  }
  //if(selected.some(Boolean)){
  //  $('.gallerybutton').text('Scatter');
  //}
  //else {
  //  $('.gallerybutton').text('Get More Images');
 // }
  image_set = []
  var cur_img_idx = 0;
  if(selected.filter(Boolean).length == 0){
    for (let i = 0; i < Object.keys(resp).length-1; i++) {      
      cur_cluster = resp[i]
      for (let j = 0; j < cur_cluster[2].length; j++) {
        $('.cex'+(j+1+cur_img_idx)).attr('src', cur_cluster[2][j]);
        $('.cex'+(j+1+cur_img_idx)).show();
        image_set.push(cur_cluster[2][j])
      }
      cur_img_idx += cur_cluster[2].length
      
    }
  }
}

$('.gallerybutton').click(function (e) {
    number_of_clicks += 1;
    var selected = []
    for (let step = 1; step < 11; step = step + 1) {
      selected.push($('.gather_c' + step).hasClass('selected'))
    }
    if(selected.filter(Boolean).length > 0) {
      for (let step = 1; step < 11; step = step + 1) {
        if(!($('.gather_c' + step).hasClass('selected'))){
          $('.gather_c' + step).hide()
        }
      }
      runPyScript(selected)
    }
});

$('.fakebutton').click(function (e) {
  get_more_images()
});

function save_direction(input){
  console.log('das');
  var jqXHR = $.ajax({
      type: "POST",
      url: flask_url + "save_direction",
      data: { mydata: JSON.stringify(input) },
      success: saved_direction_finish,
  });
} 
function saved_direction_finish(response) {
}
$('.savebutton').click(function (e) {
  save_direction(cur_direction)
  chosen_history.push(JSON.parse(JSON.stringify(cur_direction)))
  console.log(chosen_history)
  for (let step = 0; step < chosen_history.length; step = step + 1) {
    $('.chosen' + (step+1)).show();
    $('.chosen' + (step+1)).attr('src', chosen_history[step][1]);
  }
});



function go_back(input){
  var jqXHR = $.ajax({
      type: "POST",
      url: flask_url + "go_back",
      data: { mydata: JSON.stringify(input) },
      success: callbackFunc,
  });
} 
function go_next(input){
  var jqXHR = $.ajax({
      type: "POST",
      url: flask_url + "go_next",
      data: { mydata: JSON.stringify(input) },
      success: callbackFunc,
  });
} 

$('.backbutton').click(function(){
  go_back()
});


for (let step = 1; step < 6; step++) {
  $('.nextbutton' + step).click(function(){
    go_next(step-1)
  });
}