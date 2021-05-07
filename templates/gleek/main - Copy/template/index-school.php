<link rel="icon" type="image/png" sizes="16x16" href="<?= base_url() ?>public/assets/images/favicon.png">
    <!-- Custom Stylesheet -->
    <link href="<?= base_url() ?>public/css/style.css" rel="stylesheet">


    
    <!--**********************************
        Main wrapper start

        <!--**********************************
            Content body start
        ***********************************-->
        <div class="content-body">
            <div class="container-fluid">
                <div class="row page-titles">
                    <div class="col p-md-0">
                        <h4>School</h4>
                    </div>
                    <div class="col p-md-0">
                        <ol class="breadcrumb">
                            <li class="breadcrumb-item"><a href="javascript:void(0)">Home</a>
                            </li>
                            <li class="breadcrumb-item active">School</li>
                        </ol>
                    </div>
                </div>

                <div class="row">
                    <div class="col-lg-12 col-xxl-12 col-xl-5">
                        <div class="row">
                            <div class="col-sm-6 col-lg-4 col-xl-6 col-xxl-4">
                                <div class="card">
                                    <div class="card-body widget-school-stat bg-1 rounded">
                                        <div class="text">
                                            <h2>2584</h2>
                                            <p>Students</p>
                                        </div>
                                        <div class="icon">
                                            <span><i class="fa fa-user"></i></span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-sm-6 col-lg-4 col-xl-6 col-xxl-4">
                                <div class="card">
                                    <div class="card-body widget-school-stat bg-2 rounded">
                                        <div class="text">
                                            <h2>42</h2>
                                            <p>Teachers</p>
                                        </div>
                                        <div class="icon">
                                            <span><i class="fa fa-users"></i></span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-sm-6 col-lg-4 col-xl-6 col-xxl-4">
                                <div class="card">
                                    <div class="card-body widget-school-stat bg-3 rounded">
                                        <div class="text">
                                            <h2>812</h2>
                                            <p>Attendance</p>
                                        </div>
                                        <div class="icon">
                                            <span><i class="fa fa-tag"></i></span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-sm-6 col-lg-4 col-xl-6 col-xxl-4">
                                <div class="card">
                                    <div class="card-body widget-school-stat bg-4 rounded">
                                        <div class="text">
                                            <h2>45</h2>
                                            <p>Course</p>
                                        </div>
                                        <div class="icon">
                                            <span><i class="fa fa-graduation-cap"></i></span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-sm-6 col-lg-4 col-xl-6 col-xxl-4">
                                <div class="card">
                                    <div class="card-body widget-school-stat bg-5 rounded">
                                        <div class="text">
                                            <h2>&#36;2145</h2>
                                            <p>Expenses</p>
                                        </div>
                                        <div class="icon">
                                            <span><i class="fa fa-money"></i></span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div class="col-sm-6 col-lg-4 col-xl-6 col-xxl-4">
                                <div class="card">
                                    <div class="card-body widget-school-stat bg-6 rounded">
                                        <div class="text">
                                            <h2>&#36;5847</h2>
                                            <p>Income</p>
                                        </div>
                                        <div class="icon">
                                            <span><i class="fa fa-university"></i></span>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="row">
                            <div class="col-xl-12 col-sm-6 col-xxl-6">
                                <div class="card">
                                    <div class="card-body">
                                        <div class="mb-4 text-right">
                                            <select class="selectpicker show-tick">
                                                <option value="" selected>Add instructor</option>
                                                <option value="">Assign instructor</option>
                                            </select>
                                        </div>
                                        <h4>Basic Information</h4>
                                        <form class="mt-4" action="index.html">
                                            <div class="row">
                                                <div class="col-lg-5">
                                                    <div class="form-group">
                                                        <input type="text" placeholder="Title*" class="form-control bg-white text-muted " required>
                                                    </div>
                                                    <div class="form-group mb-lg-0">
                                                        <input type="text" placeholder="Last Name*" class="form-control bg-white text-muted " required>
                                                    </div>
                                                </div>
                                                <div class="col-lg-7">
                                                    <div class="form-group">
                                                        <input type="text" placeholder="First Name*" class="form-control bg-white text-muted " required>
                                                    </div>
                                                    <div class="form-group mb-0">
                                                        <input type="email" placeholder="EmailID*" class="form-control bg-white text-muted " required>
                                                    </div>
                                                </div>
                                            </div>
                                            <h4 class="mb-4 pb-3 mt-5">Assign Information</h4>
                                            <h5 class="mb-5">
                                                <span class="text-primary"><i class="fa fa-plus-circle"></i></span>
                                                Assign Class
                                            </h5>
                                            <div class="text-right">
                                                <button class="btn btn-primary btn-sl-lg">Save</button>
                                            </div>
                                        </form>
                                    </div>
                                </div>
                            </div>
                            <div class="col-xl-12 col-sm-6 col-xxl-6">
                                <div class="card widget-active-class">
                                    <div class="card-body">
                                        <h4 class="card-title mb-5">Active Classes</h4>
                                        <div class="card mb-3">
                                            <div class="card-body align-items-center d-flex justify-content-between p-4">
                                                <h5 class="mb-0 item">Astronomy 101</h5>
                                                <div class="item d-flex align-items-center">
                                                    <span class="total-students text-pale-sky"><i class="fa fa-users mr-3"></i>28</span>
                                                    <span class="add-class ml-4 text-warning c-pointer"><i class="fa fa-plus-circle"></i></span>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="card mb-3">
                                            <div class="card-body align-items-center d-flex justify-content-between p-4">
                                                <h5 class="mb-0 item">Chemistry</h5>
                                                <div class="item d-flex align-items-center">
                                                    <span class="total-students text-pale-sky"><i class="fa fa-users mr-3"></i>27</span>
                                                    <span class="add-class ml-4 text-warning c-pointer"><i class="fa fa-plus-circle"></i></span>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="card mb-3">
                                            <div class="card-body align-items-center d-flex justify-content-between p-4">
                                                <h5 class="mb-0 item">Biology</h5>
                                                <div class="item d-flex align-items-center">
                                                    <span class="total-students text-pale-sky"><i class="fa fa-users mr-3"></i>28</span>
                                                    <span class="add-class ml-4 text-warning c-pointer"><i class="fa fa-plus-circle"></i></span>
                                                </div>
                                            </div>
                                        </div>
                                        <div class="card mb-1">
                                            <div class="card-body align-items-center d-flex justify-content-between p-4">
                                                <h5 class="mb-0 item">Physics</h5>
                                                <div class="item d-flex align-items-center">
                                                    <span class="total-students text-pale-sky"><i class="fa fa-users mr-3"></i>28</span>
                                                    <span class="add-class ml-4 text-warning c-pointer"><i class="fa fa-plus-circle"></i></span>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="col-xl-7 col-lg-12 col-xxl-12">
                        <div class="card">
                            <div class="card-header pb-0 d-flex justify-content-between align-items-center">
                                <h4 class="card-title">Average Students</h4>
                                <div class="form-group absolute-icon-append-widget mb-0">
                                    <span class="absolute-form-icon absolute-form-icon-append">

                                        <i class="fa fa-search" aria-hidden="true"></i>

                                    </span>
                                    <input type="text" placeholder="Search for Students" name="avstudents" id="avstudents" class="form-control bg-white">
                                </div>
                            </div>
                            <div class="card-body pt-4">
                                <div class="chart">
                                    <div id="count-avarage-students"></div>
                                </div>
                            </div>
                        </div>
                        <div class="card">
                            <div class="card-body">
                                <h4 class="card-title mb-5">Student Enrollment</h4>
                                <div class="stat-widget-search-2">
                                    <div class="form-group border">
                                        <input type="search" name="enrollment" placeholder="Search" id="enrollment" class="form-control">
                                        <span class="search-icon"><i class="fa fa-search"></i></span>
                                    </div>
                                </div>
                                <div class="form-group border-bottom mb-5 pb-5">
                                    <select class="selectpicker show-tick" data-width="auto">
                                        <option value="">8th Grade</option>
                                        <option value="">9th Grade</option>
                                    </select>
                                    <select class="selectpicker show-tick" data-width="auto">
                                        <option value="">Biology</option>
                                        <option value="">Physics</option>
                                    </select>
                                    <select class="selectpicker show-tick" data-width="auto">
                                        <option value="">AP Biology 101</option>
                                        <option value="">AP Biology 201</option>
                                    </select>
                                </div>
                                <h4 class="card-title mb-5">Student Roaster</h4>
                                <div class="form-group pb-2">
                                    <select class="selectpicker show-tick" data-width="auto">
                                        <option value="">8th Grade</option>
                                        <option value="">9th Grade</option>
                                    </select>
                                    <span class="filter-area">
                                        <span class="mr-3"><i class="fa fa-sliders" aria-hidden="true"></i></span>
                                        Filter
                                    </span>
                                </div>
                                <div class="row">
                                    <div class="col-md-4 col-sm-6">
                                        <div class="card student-roster-card">
                                            <div class="card-body p-4 d-flex justify-content-between">
                                                <div class="item">
                                                    <h5>Scpellchek, Nigel</h5>
                                                </div>
                                                <div class="item">
                                                    <i class="fa fa-id-card" aria-hidden="true"></i>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4 col-sm-6">
                                        <div class="card student-roster-card">
                                            <div class="card-body p-4 d-flex justify-content-between">
                                                <div class="item">
                                                    <h5>Jarvis, Douchebag</h5>
                                                </div>
                                                <div class="item">
                                                    <i class="fa fa-id-card" aria-hidden="true"></i>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4 col-sm-6">
                                        <div class="card student-roster-card">
                                            <div class="card-body p-4 d-flex justify-content-between">
                                                <div class="item">
                                                    <h5>Jackson Pot, Fleece</h5>
                                                </div>
                                                <div class="item">
                                                    <i class="fa fa-id-card" aria-hidden="true"></i>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4 col-sm-6">
                                        <div class="card student-roster-card">
                                            <div class="card-body p-4 d-flex justify-content-between">
                                                <div class="item">
                                                    <h5>Will Barrow, Natalya</h5>
                                                </div>
                                                <div class="item">
                                                    <i class="fa fa-id-card" aria-hidden="true"></i>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4 col-sm-6">
                                        <div class="card student-roster-card">
                                            <div class="card-body p-4 d-flex justify-content-between">
                                                <div class="item">
                                                    <h5>Archibald, Eric Widget</h5>
                                                </div>
                                                <div class="item">
                                                    <i class="fa fa-id-card" aria-hidden="true"></i>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div class="col-md-4 col-sm-6">
                                        <div class="card student-roster-card">
                                            <div class="card-body p-4 d-flex justify-content-between">
                                                <div class="item">
                                                    <h5>View More</h5>
                                                </div>
                                                <div class="item">
                                                    <i class="fa fa-id-card" aria-hidden="true"></i>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <h4 class="card-title border-top mt-5 mb-5 pt-5">Assigned Student</h4>
                                <div class="row">
                                    <div class="col-md-4 col-sm-6">
                                        <div class="card student-roster-card">
                                            <div class="card-body p-4 d-flex justify-content-between">
                                                <div class="item">
                                                    <h5>Jarvis, Douchebag</h5>
                                                </div>
                                                <div class="item">
                                                    <i class="fa fa-id-card" aria-hidden="true"></i>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="row">
                    <div class="col-lg-12">
                        <div class="card transparent-card">
                            <div class="card-body p-0">
                                <div class="table-responsive">
                                    <table class="table table-padded status-table table-responsive-fix-bgsc-big">
                                        <thead>
                                            <tr>
                                                <th style="width: 10%">No</th>
                                                <th style="width: 15%">Name</th>
                                                <th style="width: 20%">Assigned Professor</th>
                                                <th style="width: 20%">Date of Admit</th>
                                                <th style="width: 15%">Fees</th>
                                                <th>Branch</th>
                                                <th>Action</th>
                                            </tr>
                                        </thead>
                                        <tbody>
                                            <tr>
                                                <td>01</td>
                                                <td>Fleece Marigold</td>
                                                <td>Indigo Violet</td>
                                                <td class="text-muted">02/06/2018</td>
                                                <td>
                                                    <span class="label label-success py-2 px-4 f-s-12 text-white">paid</span>
                                                </td>
                                                <td>
                                                    <span class="mr-5">Science</span>
                                                </td>
                                                <td>
                                                    <span>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky" data-original-title="Edit">
                                                            <i class="fa fa-pencil"></i> 
                                                        </a>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky ml-3" data-original-title="Close">
                                                            <i class="fa fa-close"></i>
                                                        </a>
                                                    </span>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>02</td>
                                                <td>Theodore Handle</td>
                                                <td>Niles Peppertrout</td>
                                                <td class="text-muted">04/06/2018</td>
                                                <td>
                                                    <span class="label label-success py-2 px-4 f-s-12 text-white">paid</span>
                                                </td>
                                                <td>
                                                    <span class="mr-5">Commerce</span>
                                                </td>
                                                <td>
                                                    <span>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky" data-original-title="Edit">
                                                            <i class="fa fa-pencil"></i> 
                                                        </a>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky ml-3" data-original-title="Close">
                                                            <i class="fa fa-close"></i>
                                                        </a>
                                                    </span>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>03</td>
                                                <td>Archibald Northbottom</td>
                                                <td>Jackson Pot</td>
                                                <td class="text-muted">08/06/2018</td>
                                                <td>
                                                    <span class="label label-warning py-2 px-4 f-s-12 text-white">Unpaid</span>
                                                </td>
                                                <td>
                                                    <span class="mr-5">Machanical</span>
                                                </td>
                                                <td>
                                                    <span>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky" data-original-title="Edit">
                                                            <i class="fa fa-pencil"></i> 
                                                        </a>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky ml-3" data-original-title="Close">
                                                            <i class="fa fa-close"></i>
                                                        </a>
                                                    </span>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>04</td>
                                                <td>Carnegie Mondover</td>
                                                <td>Samuel Serif</td>
                                                <td class="text-muted">12/06/2018</td>
                                                <td>
                                                    <span class="label label-warning py-2 px-4 f-s-12 text-white">Unpaid</span>
                                                </td>
                                                <td>
                                                    <span class="mr-5">Science</span>
                                                </td>
                                                <td>
                                                    <span>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky" data-original-title="Edit">
                                                            <i class="fa fa-pencil"></i> 
                                                        </a>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky ml-3" data-original-title="Close">
                                                            <i class="fa fa-close"></i>
                                                        </a>
                                                    </span>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>05</td>
                                                <td>Ursula Gurnmeister</td>
                                                <td>Fletch Skinner</td>
                                                <td class="text-muted">15/06/2018</td>
                                                <td>
                                                    <span class="label label-success py-2 px-4 f-s-12 text-white">Paid</span>
                                                </td>
                                                <td>
                                                    <span class="mr-5">M.B.A</span>
                                                </td>
                                                <td>
                                                    <span>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky" data-original-title="Edit">
                                                            <i class="fa fa-pencil"></i> 
                                                        </a>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky ml-3" data-original-title="Close">
                                                            <i class="fa fa-close"></i>
                                                        </a>
                                                    </span>
                                                </td>
                                            </tr>
                                            <tr>
                                                <td>06</td>
                                                <td>Ingredia Nutrisha</td>
                                                <td>Valentino Morose</td>
                                                <td class="text-muted">18/06/2018</td>
                                                <td>
                                                    <span class="label label-warning py-2 px-4 f-s-12 text-white">Unpaid</span>
                                                </td>
                                                <td>
                                                    <span class="mr-5">Machanical</span>
                                                </td>
                                                <td>
                                                    <span>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky" data-original-title="Edit">
                                                            <i class="fa fa-pencil"></i> 
                                                        </a>
                                                        <a href="javascript:void()" data-toggle="tooltip" data-placement="top" title=""  class="text-pale-sky ml-3" data-original-title="Close">
                                                            <i class="fa fa-close"></i>
                                                        </a>
                                                    </span>
                                                </td>
                                            </tr>
                                        </tbody>
                                    </table>
                                </div>
                                <nav>
                                    <ul class="pagination pagination-md pagination-rounded justify-content-end">
                                        <li class="page-item"><a class="page-link color-muted" href="javascript:void()">01</a>
                                        </li>
                                        <li class="page-item"><a class="page-link color-muted" href="javascript:void()">02</a>
                                        </li>
                                        <li class="page-item active"><a class="page-link color-muted" href="javascript:void()">03</a>
                                        </li>
                                        <li class="page-item"><a class="page-link color-muted" href="javascript:void()">04</a>
                                        </li>
                                        <li class="page-item"><a class="page-link color-muted" href="javascript:void()">05</a>
                                        </li>
                                    </ul>
                                </nav>
                            </div>
                        </div>
                    </div>
                </div>



            </div>
            <!-- #/ container -->
        </div>

    <!--**********************************
        Main wrapper end
    ***********************************-->

    <!--**********************************
        Scripts
    ***********************************-->

    <script src="<?= base_url() ?>public/plugins/notify/notify.min.js"></script>
    <script src="<?= base_url() ?>public/assets/plugins/common/common.min.js"></script>
    <script src="<?= base_url() ?>public/js/custom.min.js"></script>
    <script src="<?= base_url() ?>public/js/settings.js"></script>
    <script src="<?= base_url() ?>public/js/gleek.js"></script>
    <script src="<?= base_url() ?>public/js/styleSwitcher.js"></script>
    
    <script src="<?= base_url() ?>public/assets/plugins/raphael/raphael.min.js"></script>
    <script src="<?= base_url() ?>public/assets/plugins/morris/morris.min.js"></script>
    <script src="<?= base_url() ?>public/js/dashboard/dashboard-18.js"></script>
