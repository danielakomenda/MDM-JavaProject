<script>
	import Router from "svelte-spa-router";
	import routes from "./routes";
</script>

<div id="app">
        <!--Navigation-Bar-->
        <nav class="navbar navbar-expand-lg bg-body-tertiary">
            <div class="container-fluid">
                <div class="collapse navbar-collapse" id="navbarTogglerDemo01">
                    <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                        <li class="nav-item">
                            <a class="nav-link" aria-current="page" href="#/">Home</a>
                        </li>
                        <li class="nav-item">
                            <a class="nav-link" aria-current="page" href="#/classifier">Classifier</a>
                        </li>
                    </ul>
                </div>
            </div>
        </nav>
	<div class="container">
        <div class="component">
		    <Router {routes} />
        </div>
	</div>
</div>

<style>
  .nav-link {
    font-size:x-large;
    color: #1b5068;
    margin-left: 30px;
    margin-right: 30px;
  }

  a:hover {
    color: rgb(6, 158, 192);
  }

  .navbar {
    background-color: rgba(248, 242, 117, 0.8);
    height: 90px;
    width:auto;
  }

</style>