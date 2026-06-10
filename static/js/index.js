window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;
var INTERACTIVE_VIEWER_DESKTOP_QUERY = '(min-width: 1024px)';
var INTERACTIVE_VIEWER_HEIGHT = '420';
var INTERACTIVE_VIEWER_SCENES = [
  { name: 'Bunny', src: './static/images/exr/bunny.exr' },
  { name: 'Wet vs. dry materials', src: './static/images/exr/water.exr' },
  { name: 'Plants and a bill', src: './static/images/exr/money.exr' },
  { name: 'Vinyl', src: './static/images/exr/vinyl.exr' },
];

var interp_images = [];
var activeMobileSceneIndex = 0;
var renderedInteractiveViewerMode = null;

function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

function createInteractiveViewer(scene, autoOrbit) {
  var viewer = document.createElement('plenoview-viewer');
  viewer.setAttribute('src', scene.src);
  viewer.setAttribute('name', scene.name);
  viewer.setAttribute('view', '3d');
  viewer.setAttribute('source-origin', 'parent');
  viewer.setAttribute('three-d-auto-orbit', autoOrbit ? 'true' : 'false');
  viewer.setAttribute('bottom-panel', 'none');
  viewer.setAttribute('height', INTERACTIVE_VIEWER_HEIGHT);
  return viewer;
}

function createInteractiveViewerItem(scene, autoOrbit) {
  var item = document.createElement('div');
  var title = document.createElement('h3');

  item.className = 'interactive-viewer-item';
  title.className = 'title is-5 has-text-centered';
  title.textContent = scene.name;

  item.appendChild(title);
  item.appendChild(createInteractiveViewer(scene, autoOrbit));

  return item;
}

function renderDesktopInteractiveViewers() {
  var grid = document.getElementById('desktop-interactive-viewer-grid');
  if (!grid) {
    return;
  }

  grid.innerHTML = '';
  for (var i = 0; i < INTERACTIVE_VIEWER_SCENES.length; i++) {
    grid.appendChild(createInteractiveViewerItem(INTERACTIVE_VIEWER_SCENES[i], true));
  }
}

function renderMobileSceneSelector() {
  var selector = document.getElementById('mobile-scene-selector');
  if (!selector) {
    return;
  }

  selector.innerHTML = '';
  for (var i = 0; i < INTERACTIVE_VIEWER_SCENES.length; i++) {
    var scene = INTERACTIVE_VIEWER_SCENES[i];
    var button = document.createElement('button');

    button.type = 'button';
    button.className = 'mobile-scene-button';
    button.textContent = scene.name;
    button.setAttribute('aria-pressed', i === activeMobileSceneIndex ? 'true' : 'false');
    button.dataset.sceneIndex = i;

    if (i === activeMobileSceneIndex) {
      button.classList.add('is-active');
    }

    button.addEventListener('click', function(event) {
      activeMobileSceneIndex = Number(event.currentTarget.dataset.sceneIndex);
      renderMobileSceneSelector();
      renderMobileInteractiveViewer();
    });

    selector.appendChild(button);
  }
}

function renderMobileInteractiveViewer() {
  var viewer = document.getElementById('mobile-scene-viewer');
  if (!viewer) {
    return;
  }

  viewer.innerHTML = '';
  viewer.appendChild(createInteractiveViewerItem(INTERACTIVE_VIEWER_SCENES[activeMobileSceneIndex], false));
}

function renderInteractiveViewers() {
  var desktopGrid = document.getElementById('desktop-interactive-viewer-grid');
  var mobileViewer = document.getElementById('mobile-scene-viewer');
  var isDesktop = window.matchMedia(INTERACTIVE_VIEWER_DESKTOP_QUERY).matches;
  var mode = isDesktop ? 'desktop' : 'mobile';

  if (!desktopGrid || !mobileViewer || mode === renderedInteractiveViewerMode) {
    return;
  }

  renderedInteractiveViewerMode = mode;

  if (isDesktop) {
    mobileViewer.innerHTML = '';
    renderDesktopInteractiveViewers();
  } else {
    desktopGrid.innerHTML = '';
    renderMobileSceneSelector();
    renderMobileInteractiveViewer();
  }
}


$(document).ready(function() {
    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();
    renderInteractiveViewers();

    var interactiveViewerMediaQuery = window.matchMedia(INTERACTIVE_VIEWER_DESKTOP_QUERY);
    if (interactiveViewerMediaQuery.addEventListener) {
      interactiveViewerMediaQuery.addEventListener('change', renderInteractiveViewers);
    } else if (interactiveViewerMediaQuery.addListener) {
      interactiveViewerMediaQuery.addListener(renderInteractiveViewers);
    }

})
