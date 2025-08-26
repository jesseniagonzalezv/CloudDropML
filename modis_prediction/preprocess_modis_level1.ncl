

procedure processFile(cloudfile : string, geolocfile : string, split : string, output : string)
begin
  print("Processing " + cloudfile)
  print("Processing " + geolocfile)

  ; Open the files
  f1 = addfile(cloudfile, "r")
  g1 = addfile(geolocfile, "r")
  
    ; Read and process data for reflectance and radiance
    data_raw_250_Ref = f1->EV_250_Aggr1km_RefSB
    data_raw_500_Ref = f1->EV_500_Aggr1km_RefSB
    data_raw_1_Ref = f1->EV_1KM_RefSB
    data_raw_1_Ems = f1->EV_1KM_Emissive
    
    ;Get valid ranges and _FillValue for each dataset
    valid_min_250 = data_raw_250_Ref@valid_range(0)  ; Minimum valid value
    valid_max_250 = data_raw_250_Ref@valid_range(1)  ; Maximum valid value
    fill_value_250 = data_raw_250_Ref@_FillValue     ; FillValue (invalid value)
    
    valid_min_500 = data_raw_500_Ref@valid_range(0)
    valid_max_500 = data_raw_500_Ref@valid_range(1)
    fill_value_500 = data_raw_500_Ref@_FillValue
    
    valid_min_1 = data_raw_1_Ref@valid_range(0)
    valid_max_1 = data_raw_1_Ref@valid_range(1)
    fill_value_1 = data_raw_1_Ref@_FillValue
    
    valid_min_ems = data_raw_1_Ems@valid_range(0)
    valid_max_ems = data_raw_1_Ems@valid_range(1)
    fill_value_ems = data_raw_1_Ems@_FillValue
    
    ; Clean invalid data for each dataset
    data_clean_250 = where((data_raw_250_Ref .ge. valid_min_250 .and. data_raw_250_Ref .le. valid_max_250) .and. (data_raw_250_Ref .ne. fill_value_250), data_raw_250_Ref, fill_value_250)
    
    data_clean_500 = where((data_raw_500_Ref .ge. valid_min_500 .and. data_raw_500_Ref .le. valid_max_500) .and. (data_raw_500_Ref .ne. fill_value_500), data_raw_500_Ref, fill_value_500)
    
    data_clean_1 = where((data_raw_1_Ref .ge. valid_min_1 .and. data_raw_1_Ref .le. valid_max_1) .and. (data_raw_1_Ref .ne. fill_value_1), data_raw_1_Ref, fill_value_1)
    
    data_clean_ems = where((data_raw_1_Ems .ge. valid_min_ems .and. data_raw_1_Ems .le. valid_max_ems) .and. (data_raw_1_Ems .ne. fill_value_ems), data_raw_1_Ems, fill_value_ems)
    ;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;   
    
    ; Scale and offset application
    ref1_2 = conform(data_raw_250_Ref, data_raw_250_Ref@reflectance_scales, 0) * (data_clean_250 - conform(data_raw_250_Ref, data_raw_250_Ref@reflectance_offsets, 0))
    
    ref3_7 = conform(data_raw_500_Ref, data_raw_500_Ref@reflectance_scales, 0) * (data_clean_500 - conform(data_raw_500_Ref, data_raw_500_Ref@reflectance_offsets, 0))
    
    ref8_19_26 = conform(data_raw_1_Ref, data_raw_1_Ref@reflectance_scales, 0) * (data_clean_1 - conform(data_raw_1_Ref, data_raw_1_Ref@reflectance_offsets, 0))
    
    rad_20_25_27_36 = conform(data_raw_1_Ems, data_raw_1_Ems@radiance_scales, 0) * (data_clean_ems - conform(data_raw_1_Ems, data_raw_1_Ems@radiance_offsets, 0))
    
    ; Band names
    band_names_1_2 = f1->EV_250_Aggr1km_RefSB@band_names
    band_names_3_7 = f1->EV_500_Aggr1km_RefSB@band_names
    band_names_8_19_26 = f1->EV_1KM_RefSB@band_names
    band_names_20_36 = f1->EV_1KM_Emissive@band_names
    
    ; Latitude and Longitude
    lat2d1 = g1->Latitude
    lon2d1 = g1->Longitude
    
    ; Define dimensions
    bands_1_2 = 2                            ; Number of bands in ref_1_2
    bands_3_7 = 5                            ; Number of bands in ref_3_7
    bands_8_19_26 = 15                       ; Number of bands in ref_8_19_26
    bands_20_36 = 16                         ; Number of bands in rad_20_36
    
    lat_dim = dimsizes(lat2d1(:, 0))         ; Latitude dimension size
    lon_dim = dimsizes(lat2d1(0, :))         ; Longitude dimension size
    
    
    ; Attach dimensions to data before saving
    lat2d1!0 = "lat"
    lat2d1!1 = "lon"
    lon2d1!0 = "lat"
    lon2d1!1 = "lon"

    Longitude = lon2d1(0,:)
    Latitude = lat2d1(:,0)
    Latitude@units= "degrees_north"
    Longitude@units= "degrees_east"             
    Latitude!0    = "Latitude"
    Longitude!0    = "Longitude"                                                       
    Latitude&Latitude  =  Latitude                                                     
    Longitude&Longitude  =  Longitude

    

    ref1_2!0 = "bands1"
    ref1_2!1 = "lat"
    ref1_2!2 = "lon"
    ;ref1_2!1 = "y"       ; these are just index names for the grid
    ;ref1_2!2 = "x"
    ref1_2&bands1 = ispan(1, bands_1_2, 1)    ; Assign band indices
    ;ref1_2&bands1 = (/1,2/)            ; Assign band indices
    ;ref1_2&lat = Latitude
    ;ref1_2&lon = Longitude
    ; Associate the 2D coordinate variables by setting the coordinates attribute.
    ; (The coordinate variable names below must match the variable names saved to the file.)
    ref1_2@coordinates = "lat lon"


    ref3_7!0 = "bands2"
    ref3_7!1 = "lat"
    ref3_7!2 = "lon"
    ;ref3_7!1 = "y"       ; these are just index names for the grid
    ;ref3_7!2 = "x"
    ;ref3_7&bands2 = ispan(3, 7, 1)            ; Assign band indices
    ref3_7&bands2 = (/3, 4, 5, 6, 7/)            ; Assign band indices
    ;ref3_7&lat = Latitude
    ;ref3_7&lon = Longitude
    ref3_7@coordinates = "lat lon"
    
    ref8_19_26!0 = "bands3"
    ref8_19_26!1 = "lat"
    ref8_19_26!2 = "lon"
    ;ref8_19_26!1 = "y"       ; these are just index names for the grid
    ;ref8_19_26!2 = "x"
    ref8_19_26&bands3 = (/8, 9, 10, 11, 12, 13, 13.5, 14, 14.5, 15, 16, 17, 18, 19, 26/)
    ;ref8_19_26&lat = Latitude
    ;ref8_19_26&lon = Longitude
    ref8_19_26@coordinates = "lat lon"
    
    rad_20_25_27_36!0 = "bands4"
    rad_20_25_27_36!1 = "lat"
    rad_20_25_27_36!2 = "lon"
    ;rad_20_25_27_36!1 = "y"       ; these are just index names for the grid
    ;rad_20_25_27_36!2 = "x"
    rad_20_25_27_36&bands4 = (/20, 21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36/)
    ;rad_20_25_27_36&lat = Latitude
    ;rad_20_25_27_36&lon = Longitude
    rad_20_25_27_36@coordinates = "lat lon"
    
    
    ; Save to NetCDF
    cdfName1 = output + split(0) + ".nc"
    system("/bin/rm -f " + cdfName1)
    ncdf1 = addfile(cdfName1, "c")
    
    ;ncdf1->lat = lat2d1(:,:)
    ;ncdf1->lon = lon2d1(:,:)


    ; Save variables with properly named dimensions
    ncdf1->ref_1_2 = ref1_2
    ncdf1->ref_1_2@long_name = "250m Reflectance Bands"
    ncdf1->ref_1_2@band_names = band_names_1_2
    
    printMinMax(ref1_2, False)
    
    ncdf1->ref_3_7 = ref3_7
    ncdf1->ref_3_7@long_name = "500m Reflectance Bands"
    ncdf1->ref_3_7@band_names = band_names_3_7
    printMinMax(ref3_7, False)
    
    ncdf1->ref_8_19_26 = ref8_19_26
    ncdf1->ref_8_19_26@long_name = "1KM Reflective Solar Bands"
    ncdf1->ref_8_19_26@band_names = band_names_8_19_26
    printMinMax(ref8_19_26, False)
    
    ncdf1->rad_20_36 = rad_20_25_27_36
    ncdf1->rad_20_36@long_name = "1KM Emissive Bands"
    ncdf1->rad_20_36@band_names = band_names_20_36
    printMinMax(rad_20_25_27_36, False)

    ;print("Size of Latitude coordinate: " + sprinti("%d", dimsizes(Latitude)))
    ;print("Size of Longitude coordinate: " + sprinti("%d", dimsizes(Longitude)))
    ;print("Dimensions of ref1_2: " + sprinti("%d", dimsizes(ref1_2)))

    ; Save latitude and longitude as 2D variables
    ncdf1->lat = lat2d1
    ncdf1->lon = lon2d1
    ncdf1->lat@units = "degrees_north"
    ncdf1->lon@units = "degrees_east"

    ncdf1->Latitude  = Latitude
    ncdf1->Longitude = Longitude      
        
    
    print("---------------------------------------- Saving file: " + cdfName1)
    
    ; Delete variables to free memory
    delete(ref1_2)
    delete(ref3_7)
    delete(ref8_19_26)
    delete(rad_20_25_27_36)
    delete(lat2d1)
    delete(lon2d1)
    
  
end

begin
  ; Read the list of matching cloud and geoloc filenames
  cloud_files = asciiread("matching_cloud_files_level1.txt", -1, "string")
  geoloc_files  = asciiread("matching_geoloc_files_level1.txt", -1, "string")

  ; Define directories and output location
  ;diri1 = "/work/bb1143/b381362/MODIS_data/cloud/"
  ;diri2 = "/work/bb1143/b381362/MODIS_data/geoloc/"
  ;output = "/work/bb1143/b381362/MODIS_data/"

  diri1 = "/scratch/b/b381362/MODIS_data/cloud/"
  diri2 = "/scratch/b/b381362/MODIS_data/geoloc/"
  output = "/scratch/b/b381362/MODIS_data/"

  ; Get the number of matching cloud files and print it
  nfils = dimsizes(cloud_files)
  print("Processing file number: " + sprinti("%d", nfils))

  ; Loop over each matching cloud file
  do i = 0, nfils - 1
  ;do i = 0, 1 - 1
      cloudfile = cloud_files(i)
      geolocfile = geoloc_files(i)
      
      print("processing: " + cloudfile + " with " + geolocfile)

      ; Extract substring from the start up to just before the ".hdf" extension
      name_file=str_split_by_length(cloudfile, 40)
      print("name_out_file: " + name_file(0) )

      ; Call your processing function (ensure processFile is defined)
      processFile(diri1 + cloudfile, diri2 + geolocfile, name_file, output)
  end do
end
